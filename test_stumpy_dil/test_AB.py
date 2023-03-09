import unittest
import numpy as np
from parameterized import parameterized

from stumpy import stump_dil
from stumpy import stump

class TestStumpDilAB(unittest.TestCase):

    #[time_series_length_A, time_series_length_A, window_length, dilation_size, motif_pair_indices_A, motif_pair_indices_B]
    @parameterized.expand([
        ["basic", 1000, 900, 30, 2, [559, 684], [219, 84]],
        ["basic_same_positions", 1000, 900, 30, 2, [559, 684], [559, 684]],
        ["dilation_4", 1000, 900, 30, 4, [559, 684], [219, 84]],
        ["dilation_7", 1000, 900, 30, 7, [559, 684], [219, 84]],
        ["window_10", 1000, 900, 10, 2, [559, 684], [219, 84]],
        ["window_3", 1000, 900, 3, 2, [559, 684], [219, 84]],
        ["window_50", 1000, 900, 50, 2, [559, 684], [219, 84]],
        ["motif_first_and_last", 1000, 900, 30, 2, [0, 1000 - 1 - (30-1)*2], [0, 900 - 1 - (30-1)*2]],
        ["motif_first_and_middle", 1000, 900, 30, 2, [0, 630], [0, 537]],
        ["motif_middle_and_last", 1000, 900, 30, 2, [434, 1000 - 1 - (30-1)*2], [300, 900 - 1 - (30-1)*2]],
    ])
    def test_stump_dil(self, name, n_A, n_B, m, d, expected_motif_pair_indices_A, expected_motif_pair_indices_B):
        """
        Tests the stump_dil method for different parameters and motif positions
        """
        np.random.seed(0)
        T_A = np.random.randint(1, 999, n_A).astype(np.float64)
        T_A = self._insert_motif(T_A, m, d, expected_motif_pair_indices_A[0], expected_motif_pair_indices_A[1])
        T_B = np.random.randint(1, 999, n_B).astype(np.float64)
        T_B = self._insert_motif(T_B, m, d, expected_motif_pair_indices_B[0], expected_motif_pair_indices_B[1])

        mp_dil = stump_dil(T_A=T_A, T_B=T_B, m = m, d=d)

        motif_idx = np.argsort(mp_dil[:, 0])[0]
        motif_nearest_neighbor_idx = mp_dil[motif_idx, 1]
        motif_pair_indices = [motif_idx, motif_nearest_neighbor_idx]
        self.assertEqual(motif_pair_indices.sort(), expected_motif_pair_indices_A.sort())


    def test_stump_dil_without_dilation(self):
        """
        Tests the stump_dil method against the stump method
        """
        np.random.seed(0)
        n_A = 30
        n_B = 20
        m = 5
        d = 1
        T_A = np.random.randint(1, 999, n_A).astype(np.float64)
        T_B = np.random.randint(1, 999, n_B).astype(np.float64)

        mp = stump(T_A=T_A, T_B=T_B, m = m)
        mp_dil = stump_dil(T_A=T_A, T_B=T_B, m = m, d = d)

        np.testing.assert_almost_equal(mp, mp_dil)


    def _insert_motif(self, T: np.array, motif_length: int, dilation_size: int, first_motif_index: int, second_motif_index: int) -> np.array:
        for motif_value in range(0, motif_length):
            T[first_motif_index] = motif_value
            T[second_motif_index] = motif_value
            first_motif_index += dilation_size
            second_motif_index += dilation_size
        return T

if __name__ == '__main__':
    unittest.main()
