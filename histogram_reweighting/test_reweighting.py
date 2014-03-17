import unittest
from itertools import izip
import numpy as np

import matplotlib.pyplot as plt

from histogram_reweighting1d import wham1d

class HarmonicOscillator(object):
    def __init__(self, d, T):
        self.d = d
    
    def random_energy(self, T):
        x = np.random.normal(size=self.d)
        x2 = x.dot(x)
        E = T * x2 / 2
        return E

    def random_energies(self, N, T):
        x = np.random.normal(size=self.d * N)
        x = x.reshape([N,self.d])
        x2 = np.sum(x**2, axis=1)
        assert x2.size == N
        Elist = x2 * T / 2
        return Elist

    def random_histogram(self, N, binenergy, T):
        Elist = self.random_energies(N, T)
    
        evar = np.var(Elist)
        print "Cv computed from energy variance", evar / T**2 + float(self.d)/2
        
        binenergy = np.append(binenergy, binenergy[-1]+(binenergy[-1] - binenergy[-2]))
        
        counts, bins = np.histogram(Elist, binenergy)
        
        if True:
            e = binenergy[:-1]
#            plt.plot(e, counts)
            plt.plot(e, np.log(counts) + e / T )
        return counts
    
    def random_visits(self, Tlist, binenergy, N):
#        binenergy = np.linspace(0, 40, 1000)
#        visits = np.zeros([len(Tlist), len(binenergy)])
        visits = []
        for i, T in enumerate(Tlist):
            counts = self.random_histogram(N, binenergy, T)
            visits.append(counts)
#        plt.show()

        if False:
            plt.clf()
            plt.title("before numpy array")
            for T, counts in izip(Tlist, visits):
                log_nE = np.log(counts) + binenergy / T
                plt.plot(binenergy, log_nE)
            plt.show()

        visits = np.array(visits)


        if False:
            plt.clf()
            print visits.shape, binenergy.shape, Tlist.shape
            log_nET = np.log(visits) + binenergy[np.newaxis, :] / Tlist[:,np.newaxis] #+ wham.w_i_final[:,np.newaxis]
            plt.plot(binenergy, np.transpose(log_nET))
            plt.show()
#        raise Exception

        
        return visits


class TestHistogramReweighting(unittest.TestCase):
    def test_harmonic_oscilator(self):
        d = 3
        T = 1.6
        N = 100000
        Tlist = [2.000000000000000111e-01,
                2.691800385264712658e-01,
                3.622894657055626966e-01,
                4.876054616817901977e-01,
                6.562682848061104357e-01,
                8.832716109390498227e-01,
                1.188795431309558781e+00,
                1.600000000000000089e+00]
        Tlist = np.array(Tlist)
        binenergy = np.linspace(0, 20, 1000)
        ho = HarmonicOscillator(d, T)
        visits = ho.random_visits(Tlist, binenergy, N)
        assert visits.shape == (len(Tlist), len(binenergy))
        if False:
            plt.clf()
            print visits.shape, binenergy.shape, Tlist.shape
            log_nET = np.log(visits) + binenergy[np.newaxis, :] / Tlist[:,np.newaxis] #+ wham.w_i_final[:,np.newaxis]
            plt.plot(binenergy, np.transpose(log_nET))
#            for T, counts in izip(Tlist, visits):
#                log_nE = np.log(counts) + binenergy / T
#                plt.plot(binenergy, log_nE)
            plt.show()
        
        wham = wham1d(Tlist, binenergy, visits.copy())
        wham.minimize()
        cvdata = wham.calc_Cv(3, TRANGE=Tlist, use_log_sum=True)
#        print cvdata.shape
#        print cvdata
#        print cvdata
#        print "Cv values", cvdata[:,5]
        
        for cv in cvdata[:,5]:
            self.assertAlmostEqual(cv, 3, delta=.1)
        
        
        if False:
            plt.clf()
            plt.plot(binenergy, wham.logn_E)
            plt.show()
        if False:
            plt.clf()
            for r, T in enumerate(Tlist):
                v = visits[r,:]
    #            plot(bin,v)
            plt.plot(binenergy, np.log(np.transpose(visits)))
            plt.show()
        if False:
            plt.clf()
            log_nET = np.log(visits) + binenergy[np.newaxis, :]  / Tlist[:,np.newaxis] + wham.w_i_final[:,np.newaxis]
            nET = visits * np.exp(binenergy[np.newaxis, :]  / Tlist[:,np.newaxis])
            plt.plot(binenergy, np.transpose(log_nET))
            plt.plot(binenergy, wham.logn_E)
            plt.show()
        
            

if __name__ == "__main__":
    unittest.main()