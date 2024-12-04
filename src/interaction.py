import numpy as np

import brillouin_zone as bz
from i_have_channel import IHaveChannel, Channel
from i_have_mat import IHaveMat


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


class LocalInteraction(IHaveMat, IHaveChannel):
    def __init__(self, mat: np.ndarray, ur_orbs: np.ndarray, channel: Channel = Channel.NONE):
        IHaveMat.__init__(self, mat)
        IHaveChannel.__init__(self, channel)
        self._ur_orbs = ur_orbs

    def as_channel(self, channel: Channel = Channel.DENS):
        copy = LocalInteraction(self.mat, self._ur_orbs, channel)
        if copy.channel == Channel.MAGN:
            copy.mat *= -1
        elif copy.channel != Channel.DENS:
            raise ValueError(f"Invalid channel: {channel}.")
        return copy

    def __add__(self, other):
        if not isinstance(other, LocalInteraction):
            raise ValueError(f"Addition {type(self)} + {type(other)} not supported.")
        if self.channel != other.channel:
            raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
        return LocalInteraction(self.mat + other.mat, self._ur_orbs, self.channel)

    def __sub__(self, other):
        if not isinstance(other, LocalInteraction):
            raise ValueError(f"Subtraction {type(self)} - {type(other)} not supported.")
        if self.channel != other.channel:
            raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
        return LocalInteraction(self.mat - other.mat, self._ur_orbs, self.channel)


class NonLocalInteraction(LocalInteraction):
    def __init__(
        self,
        mat: np.ndarray,
        ur_r_grid: np.ndarray,
        ur_r_weights: np.ndarray,
        ur_orbs: np.ndarray,
        channel: Channel = Channel.NONE,
    ):
        super().__init__(mat, ur_orbs, channel)
        self._ur_r_grid = ur_r_grid
        self._ur_r_weights = ur_r_weights

    def get_uq(self, q_grid: bz.KGrid) -> np.ndarray:
        uk = self._convham_4_orbs(q_grid.kmesh.reshape(3, -1))
        n_bands = uk.shape[-1]
        return uk.reshape(*q_grid.nk + (n_bands,) * 4)

    def _convham_4_orbs(self, k_mesh: np.ndarray) -> np.ndarray:
        fft_grid = np.exp(1j * np.matmul(self._ur_r_grid, k_mesh)) / self._ur_r_weights[:, None, None, None, None]
        return np.transpose(np.sum(fft_grid * self.mat[..., None], axis=0), axes=(4, 0, 1, 2, 3))

    def __add__(self, other: "NonLocalInteraction"):
        if not isinstance(other, NonLocalInteraction):
            raise ValueError(f"Addition {type(self)} + {type(other)} not supported.")
        if self.channel != other.channel:
            raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
        return NonLocalInteraction(
            self.mat + other.mat, self._ur_r_grid, self._ur_r_weights, self._ur_orbs, self.channel
        )

    def __sub__(self, other: "NonLocalInteraction"):
        if not isinstance(other, NonLocalInteraction):
            raise ValueError(f"Subtraction {type(self)} - {type(other)} not supported.")
        if self.channel != other.channel:
            raise ValueError(f"Channels {self.channel} and {other.channel} don't match.")
        return NonLocalInteraction(
            self.mat - other.mat, self._ur_r_grid, self._ur_r_weights, self._ur_orbs, self.channel
        )
