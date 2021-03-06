from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit, jitclass, prange
import numpy as np
from numpy import sqrt, zeros, zeros_like, empty

node_type = deferred_type()

spec = [
    ('bounds', float64[:,:]),
    ('size', float64),
    ('delta', float64),
    ('points', float64[:,:]),
    ('masses', float64[:]),
    ('Npoints', int64),
    ('mass', float64),
    ('COM', float64[:]),
    ('IsLeaf', boolean),
    ('HasLeft', boolean),
    ('HasRight', boolean),
    ('left', optional(node_type)),
    ('right', optional(node_type)),
]



@jitclass(spec)
class KDNode(object):
    def __init__(self, points, masses):
        self.bounds = empty((3,2))
        self.bounds[0,0] = points[:,0].min()
        self.bounds[0,1] = points[:,0].max()
        self.bounds[1,0] = points[:,1].min()
        self.bounds[1,1] = points[:,1].max()
        self.bounds[2,0] = points[:,2].min()
        self.bounds[2,1] = points[:,2].max()
        
        self.size = max(self.bounds[0,1]-self.bounds[0,0],self.bounds[1,1]-self.bounds[1,0],self.bounds[2,1]-self.bounds[2,0])
        self.points = points
        self.Npoints = points.shape[0]
        self.masses = masses
        self.mass = np.sum(masses)
        self.delta = 0.
        if self.Npoints == 1:
            self.IsLeaf = True
            self.COM = points[0]
        else:
            self.IsLeaf = False
            self.COM = zeros(3)
            for k in range(3):
                for i in range(self.Npoints):
                    self.COM[k] += points[i,k]*masses[i]
                self.COM[k] /= self.mass
                self.delta += (0.5*(self.bounds[k,1]+self.bounds[k,0]) - self.COM[k])**2
            self.delta = sqrt(self.delta)
           
        self.HasLeft = False
        self.HasRight = False        
        self.left = None
        self.right = None

    def GenerateChildren(self, axis):
        if self.IsLeaf:
            return False
        x = self.points[:,axis]
        med = (self.bounds[axis,0] + self.bounds[axis,1])/2
        index = (x<med)

        if np.any(index):
            self.left = KDNode(self.points[index], self.masses[index])
            self.HasLeft = True
        index = np.invert(index)
        if np.any(index):
            self.right = KDNode(self.points[index],self.masses[index])
            self.HasRight = True
        self.points = zeros((1,1))
        self.masses = zeros(1)   
        return True

node_type.define(KDNode.class_type.instance_type)

@njit
def PotentialWalk(x, phi, node, theta=0.7):                
    r = sqrt((x[0]-node.COM[0])**2 + (x[1]-node.COM[1])**2 + (x[2]-node.COM[2])**2)
    if r>0:
        if node.size/r < theta or node.IsLeaf:
            phi -= node.mass / r
        else:
            if node.HasLeft:
                phi = PotentialWalk(x, phi, node.left, theta)
            if node.HasRight:
                phi = PotentialWalk(x, phi, node.right,  theta)
    return phi

@njit
def ForceWalk(x, g, node, theta=0.7):
    dx = node.COM[0]-x[0]
    dy = node.COM[1]-x[1]
    dz = node.COM[2]-x[2]
    r = sqrt(dx*dx + dy*dy + dz*dz)
    if r>0:
        if node.IsLeaf or r > node.size/theta + node.delta:
            r = sqrt(r*r + 0.01)
            mr3inv = node.mass/(r*r*r)
            g[0] += dx*mr3inv
            g[1] += dy*mr3inv
            g[2] += dz*mr3inv
        else:
            if node.HasLeft:
                g = ForceWalk(x, g, node.left, theta)
            if node.HasRight:
                g = ForceWalk(x, g, node.right, theta)
    return g

@njit
def CorrelationWalk(counts, rbins, x, node):
    #idea: if the center of the node is in a bin and the bounds also lie in the same bin, add to that bin. If all bounds are outside all bins, return 0. Else,repeat for children
    dx = 0.5*(node.bounds[0,0]+node.bounds[0,1])-x[0]
    dy = 0.5*(node.bounds[1,0]+node.bounds[1,1])-x[1]
    dz = 0.5*(node.bounds[2,0]+node.bounds[2,1])-x[2]
    r = (dx**2 + dy**2 + dz**2)**0.5

    sizebound = node.size*1.73
    rmin, rmax = r-sizebound/2, r+sizebound/2
    if rmin > rbins[-1]:
        return
    if rmax < rbins[0]:
        return

    N = rbins.shape[0]

    for i in range(1,N):
        if rbins[i] > r: break
        
    if rbins[i] > rmax and rbins[i-1] < rmin:
        counts[i-1] += node.Npoints
    else:
        if node.HasLeft:
            CorrelationWalk(counts, rbins, x, node.left)
        if node.HasRight:
            CorrelationWalk(counts, rbins, x, node.right)
    return

@jit
def ConstructKDTree(x, m):
    root = KDNode(x, m)
    
    nodes = np.array([root,],dtype=KDNode)
    new_nodes = empty(2,dtype=KDNode)
    axis = 0
    divisible_nodes = True
    while divisible_nodes:
        N = len(nodes)
        divisible_nodes = False
        count = 0
        for i in range(N):
            if nodes[i].IsLeaf:
                continue
            else:
                divisible_nodes += nodes[i].GenerateChildren(axis)
                if nodes[i].HasLeft:
                    new_nodes[count] = nodes[i].left
                    count += 1
                if nodes[i].HasRight:
                    new_nodes[count] = nodes[i].right
                    count += 1
                    
        axis = (axis+1)%3
        if divisible_nodes:
            nodes = new_nodes[:count]
            new_nodes = empty(count*2, dtype=KDNode)
    return root
    
@njit(parallel=True)
def GetPotentialParallel(x,tree, G, theta):
    result = empty(x.shape[0])
    for i in prange(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree,theta)
    return result

@njit
def GetPotential(x,tree, G, theta):
    result = empty(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree, theta)
    return result

@njit
def GetAccel(x, tree, G, theta):
    result = empty(x.shape)
    for i in range(x.shape[0]):
        result[i] = G*ForceWalk(x[i], zeros(3), tree, theta)
    return result

@njit(parallel=True)
def GetAccelParallel(x, tree, G, theta):
    result = empty(x.shape)
    for i in prange(x.shape[0]):
        result[i] = G*ForceWalk(x[i], zeros(3), tree, theta)
    return result

def Potential(x, m, G=1., theta=1., parallel=False):
    """Returns the approximate gravitational potential for a set of particles with positions x and masses m.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    theta -- cell opening angle used to control force accuracy; smaller is faster but more accurate. (default 1.0, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over all available cores. (default False)
    """
    tree = ConstructKDTree(np.float64(x),np.float64(m))
    result = zeros(len(m))
    if parallel:
        return GetPotentialParallel(np.float64(x),tree,G,theta)
    else:
        return GetPotential(np.float64(x),tree,G,theta)

def Accel(x, m, G=1., theta=1., parallel=False):
    tree = ConstructKDTree(np.float64(x),np.float64(m))
    result = zeros_like(x)
    if parallel:
        return GetAccelParallel(np.float64(x), tree, G, theta)
    else:
        return GetAccel(np.float64(x), tree, G, theta)

@jit
def CorrelationFunction(x, m, rbins, frac=1.):
    N = len(x)
    tree = ConstructKDTree(np.float64(x), np.float64(m))
    counts = zeros(len(rbins)-1, dtype=np.int64)
    for i in range(N):
        if np.random.rand() < frac:
            CorrelationWalk(counts, rbins, np.float64(x[i]), tree)

    return counts / (4*np.pi/3 * np.diff(rbins**3)) / frac

@njit
def BruteForcePotential(x,m,G=1.):
    potential = zeros_like(m)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[i,0]-x[j,0]
            dy = x[i,1]-x[j,1]
            dz = x[i,2]-x[j,2]
            rinv = 1./sqrt(dx*dx + dy*dy + dz*dz)
            potential[i] += m[j]*rinv
            potential[j] += m[i]*rinv
    return -G*potential

@njit#(parallel=True)
def BruteForceAccel(x,m,G=1.):
    accel = zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(i+1,x.shape[0]):
            dx = x[j,0]-x[i,0]
            dy = x[j,1]-x[i,1]
            dz = x[j,2]-x[i,2]
            r3inv = (1./sqrt(0.01 + dx*dx + dy*dy + dz*dz))**3
            accel[i,0] += m[j]*dx*r3inv
            accel[i,1] += m[j]*dy*r3inv
            accel[i,2] += m[j]*dz*r3inv
            accel[j,0] -= m[i]*dx*r3inv
            accel[j,1] -= m[i]*dy*r3inv
            accel[j,2] -= m[i]*dz*r3inv
    return G*accel

