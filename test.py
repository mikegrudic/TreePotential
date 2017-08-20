from TreePotential import *
#rom Potential import *
import numpy as np
from time import time
from matplotlib import pyplot as plt
parallel = True
N = 2**np.arange(4,21)
t1 = []
t2 = []
error = []
x = np.random.rand(10,3)
m = np.random.rand(10)
Accel(x,m,parallel=parallel)
BruteForceAccel(x,m)
Potential(x,m, parallel=parallel)
BruteForcePotential(x,m)
for n in N:
    print(n)
    x = np.random.rand(n)
    r = np.sqrt( x**(2./3) * (1+x**(2./3) + x**(4./3))/(1-x**2))
    x = np.random.normal(size=(n,3))
    x = (x.T * r/np.sum(x**2,axis=1)**0.5).T
    m = np.repeat(1./N,N) #np.random.rand(n)
    t = time()
    phi2 = Accel(x,m,parallel=parallel,theta=.7)
    #phi2 = Potential(x,m, parallel=parallel, theta=0.7)
    t2.append(time()-t)
    if n <64**3/2:
        t = time()
        phi1 = BruteForceAccel(x,m)
   #     phi1 = BruteForcePotential(x,m)
        t1.append(time()-t)
        amag = ((np.sum(phi1**2,axis=1) + np.sum(phi2**2,axis=1))/2)**0.5
        aerror = np.sum((phi1-phi2)**2,axis=1)**0.5
        #print(phi2, phi1)
        #plt.hist(np.log10(aerror/amag),100); plt.show()
        print(r[(aerror/amag).argmax()])
        error.append((aerror/amag).max())
        print(error[-1])
    else:
        t1.append(0)
        error.append(0)

print(N, t2)
plt.loglog(N, np.array(t1)/N,label="Brute Force")
plt.loglog(N, np.array(t2)/N,label="Tree")
plt.ylabel("Time per particle")
plt.xlabel("N")
plt.savefig("CPU_Time.png")
plt.clf()
plt.loglog(N, error)
plt.savefig("Errors.png")

