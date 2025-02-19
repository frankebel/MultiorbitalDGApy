import brillouin_zone as bz
from n_point_base import *
import config


class LocalInteraction(IHaveMat, IHaveChannel):
    def __init__(self, mat: np.ndarray, channel: Channel = Channel.NONE):
        IHaveMat.__init__(self, mat)
        IHaveChannel.__init__(self, channel)

    @property
    def n_bands(self) -> int:
        return self.mat.shape[0]

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "LocalInteraction":
        return LocalInteraction(np.einsum(permutation, self.mat), self.channel)

    def as_channel(self, channel: Channel) -> "LocalInteraction":
        """
        Returns the spin combination for a given channel ~WITHOUT~ the factor of 1/beta^2.
        """
        self._channel = channel
        if channel == Channel.DENS:
            return 2 * self - self.permute_orbitals("abcd->adcb")
        elif channel == Channel.MAGN:
            return -self.permute_orbitals("abcd->adcb")
        elif channel == Channel.SING:
            return self + self.permute_orbitals("abcd->adcb")
        elif channel == Channel.TRIP:
            return self - self.permute_orbitals("abcd->adcb")
        else:
            raise ValueError(f"Channel {channel} not supported.")

    def __add__(self, other) -> "LocalInteraction":
        if not isinstance(other, LocalInteraction):
            raise ValueError(f"Addition {type(self)} + {type(other)} not supported.")
        return LocalInteraction(self.mat + other.mat, self.channel)

    def __sub__(self, other) -> "LocalInteraction":
        if not isinstance(other, LocalInteraction):
            raise ValueError(f"Subtraction {type(self)} - {type(other)} not supported.")
        return LocalInteraction(self.mat - other.mat, self.channel)


class NonLocalInteraction(LocalInteraction, IAmNonLocal):
    def __init__(
        self,
        mat: np.ndarray,
        channel: Channel = Channel.NONE,
    ):
        LocalInteraction.__init__(self, mat, channel)
        IAmNonLocal.__init__(self, config.lattice.nq, config.lattice.nk, 1, 2)

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "NonLocalInteraction":
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        permutation = f"...{split[0]}->...{split[1]}"
        return NonLocalInteraction(np.einsum(permutation, self.mat), self.channel)

    def __add__(self, other: "NonLocalInteraction") -> "NonLocalInteraction":
        if not isinstance(other, (LocalInteraction, NonLocalInteraction)):
            raise ValueError(f"Addition {type(self)} + {type(other)} not supported.")

        if isinstance(other, LocalInteraction):
            other.mat = other.mat[None, None, None, ...]

        return NonLocalInteraction(
            self.mat + other.mat, self.channel if self.channel != Channel.NONE else other.channel
        )

    def __sub__(self, other: "NonLocalInteraction") -> "NonLocalInteraction":
        if not isinstance(other, (LocalInteraction, NonLocalInteraction)):
            raise ValueError(f"Subtraction {type(self)} - {type(other)} not supported.")

        if isinstance(other, LocalInteraction):
            other.mat = other.mat[None, None, None, ...]

        return NonLocalInteraction(
            self.mat - other.mat, self.channel if self.channel != Channel.NONE else other.channel
        )
