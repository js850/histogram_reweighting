import numpy as np
from numpy import log, exp
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
try:
    from scipy.optimize import Result
except ImportError:
    # they changed the name from Result to OptimizeResult at some point
    from scipy.optimize import OptimizeResult as Result



def dos_from_offsets(Tlist, binenergy, visits, offsets, nodata_value=0.):
    if visits.shape != (len(Tlist), len(binenergy)):
        raise ValueError("visits has the wrong shape")
    log_dos_all = np.where(visits==0, 0., 
                       np.log(visits) + binenergy[np.newaxis, :]  / Tlist[:,np.newaxis]
                       )
    log_dos_all = log_dos_all + offsets[:,np.newaxis]
    
    ldos = np.sum(log_dos_all * visits, axis=0)
    norm = visits.sum(0)
    ldos = np.where(norm > 0, ldos / norm, nodata_value)
    return ldos

def estimate_dos(Tlist, binenergy, visits, k_B=1.):
    """estimate the density of states from the histograms of bins
    
    Notes
    -----
    The density of states is proportional to visits * exp(E/T).  Therefore
    the log density of states for each replica is related to one another by an
    additive constant.  This function will find those offsets and produce a
    guess for the total density of states.
    
    """
    SMALL = 0.
    Tlist = Tlist * k_B
    if visits.shape != (len(Tlist), len(binenergy)):
        raise ValueError("visits has the wrong shape")
    log_dos = np.where(visits==0, SMALL, np.log(visits))
    log_dos = log_dos + binenergy[np.newaxis, :]  / Tlist[:,np.newaxis]
    
    offsets = [0.]
    for i in xrange(1,len(Tlist)):
        # find the difference in the log density of states
        ldos_diff = log_dos[i-1,:] - log_dos[i,:]
        # weight the difference by the minimum visits in each bin
        weights = np.where(visits[i-1,:] < visits[i,:], visits[i-1,:], visits[i,:])
        new_offset = np.average(ldos_diff, weights=weights)
        offsets.append( offsets[-1] + new_offset)
    offsets = np.array(offsets)
    ldos = dos_from_offsets(Tlist, binenergy, visits, offsets)

    if False:
        print offsets
        import matplotlib.pyplot as plt
        log_dos = np.where(visits==0, np.nan, log_dos)
    
        plt.clf()
        for i in xrange(len(Tlist)):
            print i
            plt.plot(binenergy, log_dos[i,:] + offsets[i])
        
        plt.plot(binenergy, ldos, 'k', lw=2)
        plt.show()
    
    return offsets, ldos


def calc_Cv(logn_E, visits1d, binenergy, NDOF, Treplica, k_B, TRANGE=None, NTEMP=100, use_log_sum = None):
    use_log_sum = True
#    if use_log_sum == None:
#        try:
#            logSumFast( np.array([.1, .2, .3]) )
#            use_log_sum = True
#            #print "using logsum"
#        except ImportError:
#            #dont use log sum unless the fast logsum is working
#            #print "not using logsum because it's too slow.  Install scipy weave to use the fast version of logsum"
#            use_log_sum = False
#    else:
#        #print "use_log_sum = ", use_log_sum
#        pass
    
    

    #put some variables in this namespace
    nrep, nebins = np.shape(visits1d)
    #print "nreps, nebins", nrep, nebins

    nz = np.where( visits1d.sum(0) != 0)[0]

    #find the ocupied bin with the minimum energy
    EREF = np.min(binenergy[nz]) - 1.

    #now calculate partition functions, energy expectation values, and Cv
    if TRANGE is None:
        #NTEMP = 100 # number of temperatures to calculate expectation values
        TMAX = Treplica[-1]
        TMIN = Treplica[0]
        TRANGE = np.linspace(TMIN, TMAX, NTEMP)
#        TINT=(TMAX-TMIN)/(NTEMP-1)
#        TRANGE = [ TMIN + i*TINT for i in range(NTEMP) ]
    NTEMP = len(TRANGE)

    dataout = np.zeros( [NTEMP, 6] )
    if abs((binenergy[-1] - binenergy[-2]) - (binenergy[-2] - binenergy[-3]) ) > 1e-7:
        print "calc_Cv: WARNING: dE is not treated correctly for exponential energy bins"

    for count,T in enumerate(TRANGE):
        kBT = k_B*T
        #find expoffset so the exponentials don't blow up
        dummy = logn_E[nz] - (binenergy[nz] - EREF)/kBT
        expoffset = np.max(dummy)
        if not use_log_sum:
            dummy = np.exp(dummy)
            Z0 = np.sum(dummy)
            Z1 = np.sum( dummy *  (binenergy[nz] - EREF) )
            Z2 = np.sum( dummy *  (binenergy[nz] - EREF)**2 )
            lZ0 = np.log(Z0)
            lZ1 = np.log(Z1)
            lZ2 = np.log(Z2)
        else:
            lZ0 = logsumexp( dummy )
            lZ1 = logsumexp( dummy + log(binenergy[nz] - EREF) )
            lZ2 = logsumexp( dummy + 2.*log(binenergy[nz] - EREF) )
            Z0 = np.exp( lZ0 )
            Z1 = np.exp( lZ1 )
            Z2 = np.exp( lZ2 )

        i = nebins-1
        #if abs(binenergy[-1] - binenergy[-2]) > 1e-7:
        #    print "calc_Cv: WARNING: dE is not treated correctly for exponential bins"
        if i == nebins-1:
            dE = binenergy[i]-binenergy[i-1]
        else:
            dE = binenergy[i+1]-binenergy[i]
            
        if dE/kBT < 1.0E-7:
            ONEMEXP=-dE/kBT
        else:
            ONEMEXP= 1.0-np.exp(dE/kBT)

        Eavg = NDOF*kBT/2.0 + 1.0*(kBT + dE/ONEMEXP) + exp(lZ1-lZ0) + EREF
        
        Cv = NDOF/2. + 1.0*(1.0 - dE**2 * exp(dE/kBT)/(ONEMEXP**2*kBT**2)) \
            - exp(lZ1 - lZ0)**2 / kBT**2 + exp(lZ2-lZ0) / kBT**2
            #- (Z1/(Z0*kBT))**2 + Z2/(Z0*kBT**2)
        
        dataout[count,0] = T
        dataout[count,1] = lZ0 + expoffset
        dataout[count,2] = lZ1 + expoffset
        dataout[count,3] = lZ2 + expoffset
        dataout[count,4] = Eavg
        dataout[count,5] = Cv

        
        #np.array([kBT, Z0*np.exp(expoffset), Z1*np.exp(expoffset), Z2*np.exp(expoffset), Eavg, Cv, np.log(Z0)+expoffset, np.log(Z1)+expoffset, np.log(Z2)+expoffset]).tofile(fout," ")
        
        #fout.write("\n")

    return dataout


def lbfgs_scipy(coords, pot, iprint=-1, tol=1e-3, nsteps=1500):
    """
    a wrapper function for lbfgs routine in scipy
    
    .. warn::
        the scipy version of lbfgs uses linesearch based only on energy
        which can make the minimization stop early.  When the step size
        is so small that the energy doesn't change to within machine precision (times the
        parameter `factr`) the routine declares success and stops.  This sounds fine, but
        if the gradient is analytical the gradient can still be not converged.  This is
        because in the vicinity of the minimum the gradient changes much more rapidly then
        the energy.  Thus we want to make factr as small as possible.  Unfortunately,
        if we make it too small the routine realizes that the linesearch routine
        isn't working and declares failure and exits.
        
        So long story short, if your tolerance is very small (< 1e-6) this routine
        will probably stop before truly reaching that tolerance.  If you reduce `factr` 
        too much to mitigate this lbfgs will stop anyway, but declare failure misleadingly.  
    """
    assert hasattr(pot, "getEnergyGradient")
    res = Result()
    res.coords, res.energy, dictionary = fmin_l_bfgs_b(pot.getEnergyGradient, 
            coords, iprint=iprint, pgtol=tol, maxfun=nsteps, factr=10.)
    res.grad = dictionary["grad"]
    res.nfev = dictionary["funcalls"]
    warnflag = dictionary['warnflag']
    #res.nsteps = dictionary['nit'] #  new in scipy version 0.12
    res.nsteps = res.nfev
    res.message = dictionary['task']
    res.success = True
    if warnflag > 0:
        print "warning: problem with quench: ",
        res.success = False
        if warnflag == 1:
            res.message = "too many function evaluations"
        else:
            res.message = str(dictionary['task'])
        print res.message
        print "    the energy is", res.energy, "the rms gradient is", np.linalg.norm(res.grad) / np.sqrt(res.grad.size), "nfev", res.nfev
        print "    X: ", res.coords
    #note: if the linesearch fails the lbfgs may fail without setting warnflag.  Check
    #tolerance exactly
    if False:
        if res.success:
            maxV = np.max( np.abs(res.grad) )
            if maxV > tol:
                print "warning: gradient seems too large", maxV, "tol =", tol, ". This is a known, but not understood issue of scipy_lbfgs"
                print res.message
    res.rms = res.grad.std()
    return res


if __name__ == "__main__":
    pass