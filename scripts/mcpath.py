import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from histogram_reweighting import wham_utils
from histogram_reweighting import wham_potential
from pele.optimize import lbfgs_py


def get_n_replicas(prefix="mcpath.f.block"):
    n = 0
    while True:
        n += 1
        if not os.path.isfile(prefix+str(n)):
            return n-1

def estimate_offsets(lprob, visits):
    offsets = [0.]
    nreps = visits.shape[0]
    for i in xrange(1,nreps):
        # find the difference in the log density of states
        lpdiff = lprob[i-1,:] - lprob[i,:]
        # weight the difference by the minimum visits in each bin
        weights = np.where(visits[i-1,:] < visits[i,:], visits[i-1,:], visits[i,:])
        new_offset = np.average(lpdiff, weights=weights)
        offsets.append( offsets[-1] + new_offset)
    return offsets


def optimize(visits, P):
    nreps, nbins = visits.shape
    P = P.copy()
    P = np.where(visits > 0, P, 0.)
    energies = np.zeros(P.shape)
    whampot = wham_potential.WhamPotential(P, energies)
    
    
    print "minimizing whampot"
    x0 = np.random.rand(nreps + nbins)
    print "initial energy", whampot.getEnergy(x0)
    ret = lbfgs_py(x0, whampot, iprint=1, tol=1e-7)
    offsets = ret.coords[:nreps]
    lp = ret.coords[nreps:]
    return lp

def main():
    prefix = "mcpath.f.block"
    nreplicas = get_n_replicas(prefix=prefix)
    print "number of replicas", nreplicas
    
    visits = []
    probabilities = []
    for i in xrange(1,nreplicas+1):
        fname = "mcpath.f.block" + str(i)
        data = np.genfromtxt(fname)
        
        
        # accumulated path length
        s = data[:,1]
        # energy of the frame
        Eframe = data[:,2]
        # probability of being in that frame
        pframe = data[:,4]
        probabilities.append(data[:,4])
        # visit probability
        visits.append(data[:,6])
    
    probabilities = np.array(probabilities)
    visits = np.array(visits)
    lprob = np.where(visits > 0, np.log(probabilities), 0.)
    lvis = np.where(visits > 0, np.log(visits), np.nan)
    
    offsets = estimate_offsets(lprob, visits)
    lprob_offset = lprob.copy()
    for i in xrange(nreplicas):
        lprob_offset[i,:] += offsets[i]
        
    lpwham = optimize(probabilities, visits)
        
    lprob_offset = np.where(visits > 0, lprob_offset, np.nan)
    lprob = np.where(visits > 0, lprob, np.nan)
    
    plt.figure(figsize=(8,10))
    plt.subplot(3,1,1)
    plt.title("log probability (with an offset applied to make neighboring regions overlap)")
    for i in xrange(nreplicas):
        plt.plot(s,lprob_offset[i,:], 'x-')
    plt.plot(s, lpwham)
        
    plt.subplot(3,1,2)
    plt.title("log probability (individually normalized, but not offset)")
    for i in xrange(nreplicas):
        plt.plot(s,lprob[i,:], 'x-')
    
    if False:
        plt.subplot(3,1,3)
        plt.title("probabilites (offset)")
        p = np.where(visits > 0, np.exp(lprob_offset), np.nan)
        for i in xrange(nreplicas):
            plt.plot(s, p[i,:], 'x-')
      
    elif False:
        plt.subplot(3,1,3)
        plt.title("log visits (individually normalized, but not offset)")
        for i in xrange(nreplicas):
            plt.plot(s,lvis[i,:], 'x-')
    
    else:
        plt.subplot(3,1,3)
        plt.plot(s, lpwham)
    
    plt.xlabel("accumulated path length")
    plt.savefig("probabilities.pdf", type="pdf")
    plt.show()
    
    
    

if __name__ == "__main__":
    main()