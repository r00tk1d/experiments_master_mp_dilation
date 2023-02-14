import unittest
import numpy as np
from parameterized import parameterized


from stumpy import stump_dil
from stumpy import stump

class TestStumpDilAA(unittest.TestCase):
    #[time_series_length, window_length, dilation_size, motif_pair_indices]
    @parameterized.expand([
        ["basic", 1000, 30, 2, [559, 684]],
        ["dilation_4", 1000, 30, 4, [559, 684]],
        ["dilation_7", 1000, 30, 7, [559, 684]],
        ["window_10", 1000, 10, 2, [559, 684]],
        ["window_3", 1000, 3, 2, [559, 684]],
        ["window_50", 1000, 50, 2, [559, 684]],
        ["motif_first_and_last", 1000, 30, 2, [0, 1000 - 1 - (30-1)*2]],
        ["motif_first_and_middle", 1000, 30, 2, [0, 630]],
        ["motif_middle_and_last", 1000, 30, 2, [434, 1000 - 1 - (30-1)*2]],
    ])
    def test_stump_dil(self, name, time_series_length, m, d, expected_motif_pair_indices):
        """
        Tests the stump_dil method for different parameters and motif positions
        """
        np.random.seed(0)
        T = np.random.randint(1, 999, time_series_length).astype(np.float64)
        T = self._insert_motif(T, m, d, expected_motif_pair_indices[0], expected_motif_pair_indices[1])

        mp_dil = stump_dil(T_A=T, m = m, d=d)

        motif_idx = np.argsort(mp_dil[:, 0])[0]
        motif_nearest_neighbor_idx = mp_dil[motif_idx, 1]
        motif_pair_indices = [motif_idx, motif_nearest_neighbor_idx]
        self.assertEqual(motif_pair_indices.sort(), expected_motif_pair_indices.sort())


    def test_stump_dil_without_dilation(self):
        """
        Tests the stump_dil method against the stump method
        """
        np.random.seed(0)
        time_series_length = 30
        m = 3
        d = 1
        T = np.random.randint(1, 999, time_series_length).astype(np.float64)

        mp = stump(T_A=T, m = m)
        mp_dil = stump_dil(T_A=T, m = m, d = d)

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
