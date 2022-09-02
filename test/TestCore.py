import unittest
import numpy as np
import Core

class TestZerothMoment(unittest.TestCase):
    def setUp(self):
        self.m = 10
        self.n = 12
        self.o = 10
        self.f =  np.zeros((self.m, self.n, self.o, 27), dtype = float)

    def test_dimensions_correct(self):
        zerothMoment = Core.zerothMoment(self.f)
        self.assertEqual(zerothMoment.shape, (self.m, self.n, self.o))

    def test_dummy(self):
        self.assertTrue(False,"Dummy test failed")

if __name__ == '__main__':
    unittest.main()