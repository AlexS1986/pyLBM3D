import unittest
import numpy as np
import Experimental
import Settings

class TestZerothMoment(unittest.TestCase):
    def setUp(self):
        # self.m = 10
        # self.n = 12
        # self.o = 10
        # self.f = np.zeros((self.m, self.n, self.o, 27), dtype = float)


        [self.cc, self.w]= Settings.getLatticeVelocitiesWeights(0.1)




    def test_feq(self):
        rhoTest = np.array(
            [[[2.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
             [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
             ], dtype=float)
        jTest = np.array([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                               [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                               [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
                               ], dtype=float)

        sTest = np.zeros((3, 3, 3, 3, 3), dtype=float)
        la = 1.5
        mue = 1.0
        rho0 = 1.0
        cs = (mue/rho0) ** 0.5
        feqOutTest = Experimental.equilibriumDistribution(rhoTest, jTest, sTest, self.cc, self.w,
                                                          cs, la, mue, rho0)
        self.assertEqual(feqOutTest.shape,(3,3,3,27),"Shape should be " + feqOutTest.shape.__str__())


    def test_dummy(self):
        self.assertTrue(True, "Dummy test failed")


    def test_einstein(self):
        I = np.identity(3, dtype=np.float)
        D = np.array([0.1,1.0,2.0], dtype=np.float)
        out = np.einsum('a,bc->abc', D,I)
        self.assertEqual(out.shape,(3,3,3))

if __name__ == '__main__':
    unittest.main()