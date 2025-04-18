import numpy as np

from scdga.local_n_point import LocalNPoint
from scdga.n_point_base import IAmNonLocal, IHaveChannel, SpinChannel, FrequencyNotation


class GapFunction(LocalNPoint, IAmNonLocal, IHaveChannel):
    """
    Represents the superconducting gap function.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)
        IHaveChannel.__init__(self, channel, FrequencyNotation.PP)

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def to_compound_indices(self):
        """
        Converts the indices of the gap function to compound indices.
        """
        if len(self.current_shape) == 2:  # [q,x]
            return self

        self.update_original_shape()
        self.mat = self.mat.reshape(self.nq_tot, self.n_bands**2 * 2 * self.niv)
        return self

    def to_full_indices(self, shape: tuple = None):
        """
        Converts an object stored with compound indices to an object that has unraveled orbital and frequency axes.
        """
        if len(self.current_shape) == (
            4 if self.has_compressed_q_dimension else 6
        ):  # [q,o1,o2,v] or [qx,qy,qz,o1,o2,v]
            return self

        self.original_shape = shape if shape is not None else self.original_shape

        if self.has_compressed_q_dimension:
            self.mat = self.mat.reshape(self.nq_tot, self.n_bands, self.n_bands, 2 * self.niv)
        else:
            self.mat = self.mat.reshape(*self.nq, self.n_bands, self.n_bands, 2 * self.niv)
        return self

    def add(self, other):
        """
        Adds two GapFunctions.
        """
        if not isinstance(other, GapFunction):
            raise TypeError("Can only add or subtract GapFunction objects.")

        self.compress_q_dimension()
        other = other.compress_q_dimension()

        return GapFunction(
            self.mat + other.mat,
            self.channel,
            nk=self.nq,
            full_niv_range=self.full_niv_range,
            has_compressed_q_dimension=True,
        )

    def sub(self, other):
        """
        Subtracts two GapFunctions.
        """
        return self.add(-other)

    def __add__(self, other):
        """
        Adds two GapFunctions.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Subtracts two GapFunctions.
        """
        return self.sub(other)
