import unittest

import numpy as np

from histogram_reweighting import wham_utils
import test_reweighting


class TestCvCalc(unittest.TestCase):
    def test(self):
        d = 3
        ho = test_reweighting.HarmonicOscillator(d)
        binenergy = np.linspace(0, 40, 200)
        Tlist = np.linspace(.2, 1.6, 8)
        log_dos = ho.make_analytic_dos(binenergy)
        
        visits = np.ones([len(Tlist), len(binenergy)]) # this is only used to see where there is no data
        cvdata = wham_utils.calc_Cv(log_dos, visits, binenergy, d, Tlist, 1.)
        cv = cvdata[:,5]
        self.assertLess(np.max(np.abs(cv - d)), 0.05)
        
if __name__ == "__main__":
    unittest.main()
