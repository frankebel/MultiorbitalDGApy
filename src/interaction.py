import numpy as np

from local_four_point import Channel
from i_have_mat import IHaveMat
import brillouin_zone as bz


class InteractionElement:
    def __init__(self, r_lat: list[int], orbs: list[int], value: float):
        if not isinstance(r_lat, list) and len(r_lat) == 3 and all(isinstance(x, int) for x in r_lat):
            raise ValueError("'r_lat' must be a list with exactly 3 integer elements.")
        if (
            not isinstance(orbs, list)
            and len(orbs) == 4
            and all(isinstance(x, int) for x in orbs)
            and all(orb > 0 for orb in orbs)
        ):
            raise ValueError("'orbs' must be a list with exactly 4 integer elements that are greater than zero.")
        if not isinstance(value, (int, float)):
            raise ValueError("'value' must be a real number.")

        self.r_lat = tuple(r_lat)
        self.orbs = np.array(orbs, dtype=int)
        self.value = float(value)


class LocalInteraction(IHaveMat):
    def __init__(self, mat: np.ndarray, ur_orbs: np.ndarray):
        super().__init__(mat)
        self._ur_orbs = ur_orbs

    def get_for_channel(self, channel: Channel):
        copy = self
        if channel == Channel.MAGN:
            copy.mat *= -1
        elif channel != Channel.DENS:
            raise ValueError(f"Invalid channel: {channel}.")
        return copy


class NonLocalInteraction(LocalInteraction):
    def __init__(self, mat: np.ndarray, ur_r_grid: np.ndarray, ur_r_weights: np.ndarray, ur_orbs: np.ndarray):
        super().__init__(mat, ur_orbs)
        self._ur_r_grid = ur_r_grid
        self._ur_r_weights = ur_r_weights

    def get_uq(self, q_grid: bz.KGrid) -> np.ndarray:
        uk = self._convham_4_orbs(q_grid.kmesh.reshape(3, -1))
        n_bands = uk.shape[-1]
        return uk.reshape(*q_grid.nk + (n_bands,) * 4)

    def _convham_4_orbs(self, k_mesh: np.ndarray) -> np.ndarray:
        fft_grid = np.exp(1j * np.matmul(self._ur_r_grid, k_mesh)) / self._ur_r_weights[:, None, None, None, None]
        return np.transpose(np.sum(fft_grid * self.mat[..., None], axis=0), axes=(4, 0, 1, 2, 3))
