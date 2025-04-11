from n_point_base import *


class LocalInteraction(IHaveMat, IHaveChannel):
    r"""
    Class for local interactions
    .. math:: U_{abcd}.
    """

    def __init__(self, mat: np.ndarray, channel: SpinChannel = SpinChannel.NONE):
        IHaveMat.__init__(self, mat)
        IHaveChannel.__init__(self, channel)

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[0]

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "LocalInteraction":
        """
        Permutes the orbitals of the object. The permutation string must be given in the einsum notation.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        return LocalInteraction(np.einsum(permutation, self.mat, optimize=True), self.channel)

    def as_channel(self, channel: SpinChannel) -> "LocalInteraction":
        """
        Returns the spin combination for a given channel.
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

    def add(self, other) -> "LocalInteraction":
        """
        Adds two local interactions.
        """
        if not isinstance(other, (LocalInteraction, np.ndarray)):
            raise ValueError(f"Operation {type(self)} +/- {type(other)} not supported.")

        if isinstance(other, np.ndarray):
            return Interaction(self.mat + other, self.channel)

        return LocalInteraction(
            self.mat + other.mat, self.channel if self.channel != SpinChannel.NONE else other.channel
        )

    def sub(self, other) -> "LocalInteraction":
        """
        Subtracts two local interactions.
        """
        return self.add(-other)

    def pow(self, power) -> "LocalInteraction":
        """
        Exponentiation of LocalInteraction objects.
        """
        if power <= 0:
            raise ValueError("Exponentiation of Interaction objects only supports positive powers greater than zero.")
        result = deepcopy(self)
        for _ in range(1, power):
            result = LocalInteraction(result.times("abcd,dcef->abef", self), self.channel)
        return result

    def __add__(self, other) -> "LocalInteraction":
        """
        Adds two local interactions.
        """
        return self.add(other)

    def __radd__(self, other) -> "LocalInteraction":
        """
        Adds two local interactions.
        """
        return self.add(other)

    def __sub__(self, other) -> "LocalInteraction":
        """
        Subtracts two local interactions.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "LocalInteraction":
        """
        Subtracts two local interactions.
        """
        return self.sub(other)

    def __pow__(self, power, modulo=None) -> "LocalInteraction":
        """
        Exponentiation of Interaction objects.
        """
        return self.pow(power)


class Interaction(LocalInteraction, IAmNonLocal):
    r"""
    Class for non-local interactions
    .. math:: V_{abcd}^{q}.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        has_compressed_q_dimension: bool = False,
    ):
        LocalInteraction.__init__(self, mat, channel)
        IAmNonLocal.__init__(self, mat, nq, has_compressed_q_dimension)

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "Interaction":
        """
        Permutes the orbitals of the object. The permutation string must be given in the einsum notation.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        permutation = f"...{split[0]}->...{split[1]}"
        return Interaction(
            np.einsum(permutation, self.mat, optimize=True), self.channel, self.nq, self.has_compressed_q_dimension
        )

    def as_channel(self, channel: SpinChannel) -> "Interaction":
        """
        Returns the spin combination for a given channel. Note that we only have the non-local ph contribution
        in the ladder DGA equations and the phbar contribution to the spin channels vanishes.
        """
        copy = deepcopy(self)

        if copy.channel == channel:
            return copy
        elif copy.channel != channel.NONE:
            raise ValueError(f"Cannot transform interaction from channel {copy.channel} to {channel}.")

        copy._channel = channel
        if channel == SpinChannel.DENS:
            return 2 * copy
        elif channel == SpinChannel.MAGN:
            return 0 * copy
        elif channel == SpinChannel.SING:
            return copy
        elif channel == SpinChannel.TRIP:
            return copy
        else:
            raise ValueError(f"Channel {channel} not supported.")

    def add(self, other) -> "Interaction":
        """
        Adds two (non-)local interactions.
        """
        if not isinstance(other, (LocalInteraction, Interaction, np.ndarray)):
            raise ValueError(f"Operation {type(self)} +/- {type(other)} not supported.")

        if isinstance(other, np.ndarray):
            return Interaction(self.mat + other, self.channel, self.nq)

        if not isinstance(other, Interaction):
            other_mat = other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...]
        else:
            other_mat = other.mat

        return Interaction(
            self.mat + other_mat,
            self.channel if self.channel != SpinChannel.NONE else other.channel,
            self.nq,
            self.has_compressed_q_dimension,
        )

    def sub(self, other) -> "Interaction":
        """
        Subtracts two (non-)local interactions.
        """
        return self.add(-other)

    def pow(self, power) -> "Interaction":
        """
        Exponentiation of Interaction objects.
        """
        if power <= 0:
            raise ValueError("Exponentiation of Interaction objects only supports positive powers greater than zero.")
        is_self_compressed = self.has_compressed_q_dimension
        result = deepcopy(self).compress_q_dimension()
        for _ in range(1, power):
            result = Interaction(result.times("qabcd,qdcef->qabef", self), self.channel, self.nq, True)
        return result if is_self_compressed else result.decompress_q_dimension()

    def __add__(self, other) -> "Interaction":
        """
        Adds two (non-)local interactions.
        """
        return self.add(other)

    def __radd__(self, other) -> "Interaction":
        """
        Adds two (non-)local interactions.
        """
        return self.add(other)

    def __sub__(self, other) -> "Interaction":
        """
        Subtracts two (non-)local interactions.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "Interaction":
        """
        Subtracts two (non-)local interactions.
        """
        return self.sub(other)

    def __pow__(self, power, modulo=None) -> "Interaction":
        """
        Exponentiation of Interaction objects.
        """
        return self.pow(power)
