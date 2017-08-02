from numba import jitclass, prange
from numba import int32, deferred_type, optional, float64, boolean, int64, njit, jit
import numpy as np


node_type = deferred_type()

spec = [
    ('bounds', float64[:,:]),
    ('size', float64),
    ('points', float64[:,:]),
    ('masses', float64[:]),
    ('Npoints', int64),
    ('mass', float64),
    ('COM', float64[:]),
    ('potential', float64),
    ('IsLeaf', boolean),
    ('HasLeft', boolean),
    ('HasRight', boolean),
    ('left', optional(node_type)),
    ('right', optional(node_type)),
]

@jitclass(spec)
class KDNode(object):
    def __init__(self, bounds, points, masses):
        self.bounds = bounds
        self.size = max(bounds[0,1]-bounds[0,0],bounds[1,1]-bounds[1,0],bounds[2,1]-bounds[2,0])
        self.points = points
        self.potential = 0.
        self.Npoints = points.shape[0]
        self.masses = masses
        self.mass = masses.sum()
        if self.Npoints < 2:
            self.IsLeaf = True
            self.COM = points[0]
        else:
            self.IsLeaf = False
            self.COM = np.zeros(3)
            for k in range(3):
                for i in range(self.Npoints):
                    self.COM[k] += points[i,k]*masses[i]
                self.COM[k] /= self.mass
        self.HasLeft = False
        self.HasRight = False        
        self.left = None
        self.right = None

            

node_type.define(KDNode.class_type.instance_type)

@njit
def PotentialWalk(x, phi, node, theta=0.7):
    r = ((x[0]-node.COM[0])**2 + (x[1]-node.COM[1])**2 + (x[2]-node.COM[2])**2)**0.5
    if r>0:
        if node.IsLeaf or node.size/r < theta:
            phi -= node.mass / r
        else:
            if node.HasLeft:
                phi = PotentialWalk(x, phi, node.left, theta)
            if node.HasRight:
                phi = PotentialWalk(x, phi, node.right,  theta)
    return phi

@njit
def ForceWalk(x, g, node, theta=0.7):
    dx = np.empty(3)
    for k in range(3):
        dx[k] = node.COM[k] - x[k]
    r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5
    if r:
        if node.IsLeaf or node.size/r < theta:
            for k in range(3):
                g[k] += node.mass*dx[k]/ r**3
        else:
            if node.HasLeft:
                g = ForceWalk(x, g, node.left, theta)
            if node.HasRight:
                g = ForceWalk(x, g, node.right, theta)
    return g

    
@njit
def GenerateChildren(node, axis):
    N = node.Npoints
    if N < 2:
        return False
    
    x = node.points[:,axis]

    if N < 1000:
        xsort = np.sort(x)
        med = xsort[N//2]
    else:
        med = np.median(x)

    index = (x<med)
    bounds_left = np.copy(node.bounds)
    bounds_left[axis,1] = med
    bounds_right = np.copy(node.bounds)
    bounds_right[axis,0] = med
    if np.any(index):
        node.left = KDNode(bounds_left, node.points[index], node.masses[index])
        node.HasLeft = True
    index = np.invert(index)
    if np.any(index):
        node.right = KDNode(bounds_right, node.points[index],node.masses[index])
        node.HasRight = True
    node.points = np.zeros((1,1))
    node.masses = np.zeros(1)   
    return True

@jit
def ConstructKDTree(x, m):
    xmin, ymin, zmin = np.min(x,axis=0)
    xmax, ymax, zmax = np.max(x,axis=0)
    bounds = np.empty((3,2))
    bounds[0,0] = xmin
    bounds[0,1] = xmax
    bounds[1,0] = ymin
    bounds[1,1] = ymax
    bounds[2,0] = zmin
    bounds[2,1] = zmax
    root = KDNode(bounds, x, m)
    
    nodes = np.array([root,],dtype=KDNode)
    new_nodes = np.empty(2,dtype=KDNode)
    leafnodes = []
    divisible_nodes = True
    depth = 0
    axis = 0
    while divisible_nodes:
        N = len(nodes)
        divisible_nodes = False
        count = 0
        depth += 1
        for i in range(N):
            if nodes[i].IsLeaf:
                continue
            else:
                divisible_nodes += GenerateChildren(nodes[i],axis)
                if nodes[i].HasLeft:
                    new_nodes[count] = nodes[i].left
                    count += 1
                if nodes[i].HasRight:
                    new_nodes[count] = nodes[i].right
                    count += 1
        axis = (axis+1)%3
        if divisible_nodes:
            nodes = new_nodes[:count]
            new_nodes = np.empty(count*2, dtype=KDNode)
    return root
    
@njit(parallel=True)
def GetPotentialParallel(x,tree, G, theta):
    result = np.empty(x.shape[0])
    for i in prange(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree,theta)
    return result

@njit
def GetPotential(x,tree, G, theta):
    result = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        result[i] = G*PotentialWalk(x[i],0.,tree, theta)
    return result
    

def Potential(x, m, G=1., theta=1., parallel=False):
    """Returns the approximate gravitational potential for a set of particles with positions x and masses m.

    Arguments:
    x -- shape (N,3) array of particle positions
    m -- shape (N,) array of particle masses

    Keyword arguments:
    G -- gravitational constant (default 1.0)
    theta -- cell opening angle used to control force accuracy; smaller is faster but more accurate. (default 1.0, gives ~1% accuracy)
    parallel -- If True, will parallelize the force summation over $OMP_NUM_THREADS cores. (default False)
    """
    tree = ConstructKDTree(x,m)
    result = np.zeros(len(m))
    if parallel:
        return GetPotentialParallel(x,tree,G,theta)
    else:
        return GetPotential(x,tree,G,theta)