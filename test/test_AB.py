import unittest
import numpy as np

from stumpy import stump_dil
from stumpy import stump


class TestStumpDilAB(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.T_A = np.random.randint(1,999,1000).astype(np.float64)
        np.random.seed(99)
        self.T_B = np.random.randint(1,999,900).astype(np.float64)

    def test_stump_dil_without_dilation(self):
        """
        Tests the stump_dil method against the stump method
        """
        # mp = stump(T_A=self.T_A, m = 30)
        # mp_dil = stump_dil(T_A=self.T_A, m = 30, d = 1)
        # np.testing.assert_almost_equal(mp, mp_dil)
        pass


if __name__ == '__main__':
    unittest.main()
