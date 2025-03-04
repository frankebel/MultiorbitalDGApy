import brillouin_zone as bz
from n_point_base import *
import config


class LocalInteraction(IHaveMat, IHaveChannel):
    def __init__(self, mat: np.ndarray, channel: SpinChannel = SpinChannel.NONE):
        IHaveMat.__init__(self, mat)
        IHaveChannel.__init__(self, channel)

    @property
    def n_bands(self) -> int:
        return self.mat.shape[0]

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "LocalInteraction":
        return LocalInteraction(np.einsum(permutation, self.mat), self.channel)

    def as_channel(self, channel: SpinChannel) -> "LocalInteraction":
        """
        Returns the spin combination for a given channel ~WITHOUT~ a factor of 1/beta^2.
        """
        if self.channel == channel:
            return self
        self._channel = channel
        perm: str = "abcd->adcb"
        if channel == SpinChannel.DENS:
            return 2 * self - self.permute_orbitals(perm)
        elif channel == SpinChannel.MAGN:
            return -self.permute_orbitals(perm)
        elif channel == SpinChannel.SING:
            return self + self.permute_orbitals(perm)
        elif channel == SpinChannel.TRIP:
            return self - self.permute_orbitals(perm)
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
        channel: SpinChannel = SpinChannel.NONE,
    ):
        LocalInteraction.__init__(self, mat, channel)
        IAmNonLocal.__init__(self, mat, config.lattice.nq)

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "NonLocalInteraction":
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        permutation = f"...{split[0]}->...{split[1]}"
        return NonLocalInteraction(np.einsum(permutation, self.mat), self.channel)

    def _execute_add_sub(self, other, is_addition: bool) -> "NonLocalInteraction":
        if not isinstance(other, (LocalInteraction, NonLocalInteraction)):
            raise ValueError(
                f"Addition {type(self)} + {type(other)} not supported."
                if is_addition
                else f"Subtraction {type(self)} - {type(other)} not supported."
            )

        if isinstance(other, LocalInteraction):
            other = other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...]

        return NonLocalInteraction(
            self.mat + other.mat if is_addition else self.mat - other.mat,
            self.channel if self.channel != SpinChannel.NONE else other.channel,
        )

    def __add__(self, other) -> "NonLocalInteraction":
        return self._execute_add_sub(other, True)

    def __sub__(self, other) -> "NonLocalInteraction":
        return self._execute_add_sub(other, False)
