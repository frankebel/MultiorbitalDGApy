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
        copy = deepcopy(self)

        if copy.channel == channel:
            return copy
        elif copy.channel != channel.NONE:
            raise ValueError(f"Cannot transform interaction from channel {copy.channel} to {channel}.")

        copy._channel = channel
        perm: str = "abcd->adcb"
        if channel == SpinChannel.DENS:
            return 2 * copy - copy.permute_orbitals(perm)
        elif channel == SpinChannel.MAGN:
            return -copy.permute_orbitals(perm)
        elif channel == SpinChannel.SING:
            return copy + copy.permute_orbitals(perm)
        elif channel == SpinChannel.TRIP:
            return copy - copy.permute_orbitals(perm)
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
        return NonLocalInteraction(np.einsum(permutation, self.mat, optimize=True), self.channel)

    def add(self, other) -> "NonLocalInteraction":
        if not isinstance(other, (LocalInteraction, NonLocalInteraction)):
            raise ValueError(f"Operation {type(self)} +/- {type(other)} not supported.")

        if isinstance(other, LocalInteraction):
            other = other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...]

        return NonLocalInteraction(
            self.mat + other.mat,
            self.channel if self.channel != SpinChannel.NONE else other.channel,
        )

    def __add__(self, other) -> "NonLocalInteraction":
        return self.add(other)

    def __radd__(self, other) -> "NonLocalInteraction":
        return self.add(other)

    def __sub__(self, other) -> "NonLocalInteraction":
        return self.add(-other)

    def __rsub__(self, other) -> "NonLocalInteraction":
        return self.add(-other)
