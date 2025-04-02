import gc
import os
from abc import ABC
from copy import deepcopy
from enum import Enum

import numpy as np


class SpinChannel(Enum):
    DENS: str = "dens"
    MAGN: str = "magn"
    SING: str = "sing"
    TRIP: str = "trip"
    UU: str = "uu"
    UD: str = "ud"
    UD_BAR: str = "ud_bar"
    NONE: str = "none"


class FrequencyNotation(Enum):
    PH: str = "ph"
    PH_BAR: str = "ph_bar"
    PP: str = "pp"


class IHaveChannel(ABC):
    """
    Abstract interface for classes that have a channel attribute.
    """

    def __init__(
        self, channel: SpinChannel = SpinChannel.NONE, frequency_notation: FrequencyNotation = FrequencyNotation.PH
    ):
        self._channel = channel
        self._frequency_notation = frequency_notation

    @property
    def channel(self) -> SpinChannel:
        """
        Returns the channel reducibility (not the frequency notation) of the object.
        For a set of available channels, see class Channel.
        """
        return self._channel

    @property
    def frequency_notation(self) -> FrequencyNotation:
        """
        Returns the frequency notation (not the channel reducibility) of the object.
        For a set of available notations, see class FrequencyNotation.
        """
        return self._frequency_notation


class IHaveMat(ABC):
    """
    Abstract interface for classes that have a mat attribute. Adds a couple of convenience methods for matrix operations.
    """

    def __init__(self, mat: np.ndarray):
        self.mat = mat
        self._original_shape = self.mat.shape

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the underlying matrix.
        """
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        """
        Sets the underlying matrix.
        """
        self._mat = value.astype(np.complex64)

    @property
    def current_shape(self) -> tuple:
        """
        Keeps track of the current shape of the underlying matrix.
        """
        return self._mat.shape

    @property
    def original_shape(self) -> tuple:
        """
        Keeps track of the previous shape of the underlying matrix before the reshaping process. E.g., it is needed when
        reshaping it to compound indices where the original shape would have been lost otherwise.
        """
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value) -> None:
        """
        Sets the original shape of the matrix. Keeps track of the previous shape of the underlying matrix
        before the reshaping process. E.g., it is needed when reshaping it to compound indices where the
        original shape would have been lost otherwise.
        """
        self._original_shape = value

    @property
    def memory_usage_in_gb(self) -> float:
        """
        Returns the memory usage of the matrix in GigaBytes (GB).
        """
        return self.mat.nbytes / (1024**3)

    def to_real(self) -> "IHaveMat":
        """
        Converts the matrix to real numbers. Returns it as complex type.
        """
        dtype = self.mat.dtype
        self.mat = self.mat.real.astype(dtype)
        return self

    def __mul__(self, other) -> "IHaveMat":
        """
        Multiplication with a scalar or another matrix.
        """
        if not isinstance(other, (int, float, complex, np.ndarray)):
            raise ValueError("Multiplication only supported with numbers or numpy arrays.")

        copy = deepcopy(self)
        copy.mat *= other
        return copy

    def __rmul__(self, other) -> "IHaveMat":
        """
        Right multiplication with a scalar or another matrix.
        """
        return self.__mul__(other)

    def __neg__(self) -> "IHaveMat":
        """
        Negation of the matrix.
        """
        return self.__mul__(-1.0)

    def __truediv__(self, other) -> "IHaveMat":
        """
        Division with a scalar.
        """
        if not isinstance(other, (int, float, complex)):
            raise ValueError("Division only supported with numbers.")
        return self.__mul__(1.0 / other)

    def __getitem__(self, item):
        """
        Returns the value at position [item].
        """
        return self.mat[item]

    def __setitem__(self, key, value):
        """
        Sets the value at position [key].
        """
        self.mat[key] = value

    def __del__(self):
        """
        Deletes the underlying matrix.
        """
        del self._mat
        gc.collect()

    def update_original_shape(self):
        """
        Updates the original shape of the matrix. This is needed when the matrix is reshaped.
        """
        self.original_shape = self.current_shape

    def times(self, contraction: str, *args) -> np.ndarray:
        """
        Multiplies the matrices of multiple objects with the contraction
        specified and returns the result as a numpy array.
        """
        if not all(isinstance(obj, (IHaveMat, np.ndarray)) for obj in args):
            raise ValueError("Args has atleast one object with the wrong type. Allowed are [IHaveMat] or [np.ndarray].")
        return np.einsum(
            contraction, self.mat, *[obj.mat if isinstance(obj, IHaveMat) else obj for obj in args], optimize=True
        )


class IAmNonLocal(IHaveMat, ABC):
    """
    Abstract interface for objects that are momentum dependent. Since we focus on ladder objects, we do not
    need more than one momentum variable for one- and two-particle quantities.
    """

    def __init__(self, mat: np.ndarray, nq: tuple[int, int, int], has_compressed_q_dimension: bool = False):
        super().__init__(mat)
        self._nq = nq
        self._has_compressed_q_dimension = has_compressed_q_dimension

    @property
    def nq(self) -> tuple[int, int, int]:
        """
        Returns the number of momenta in the object.
        """
        return self._nq

    @property
    def nq_tot(self) -> int:
        """
        Returns the total number of momenta in the object.
        """
        return np.prod(self.nq).astype(int) if not self.has_compressed_q_dimension else self.original_shape[0]

    @property
    def has_compressed_q_dimension(self) -> bool:
        """
        Returns whether the underlying matrix has a compressed momentum dimension (q,...) or not (qx,qy,qz,...).
        """
        return self._has_compressed_q_dimension

    def shift_k_by_q(self, index: tuple | list[int] = (0, 0, 0)):
        """
        Shifts the momentum by the given value. If the object enters with a compressed q dimension, we first decompress
        it and then compress it again after the shift.
        """
        copy = deepcopy(self)

        compressed = False
        if copy.has_compressed_q_dimension:
            copy = copy.decompress_q_dimension()
            compressed = True

        copy.mat = np.roll(copy.mat, index, axis=(0, 1, 2))
        return copy if not compressed else copy.compress_q_dimension()

    def compress_q_dimension(self):
        """
        Converts the object from (qx,qy,qz,...) to (q,...) in-place, where len(q) = qx*qy*qz.
        """
        if self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((self.nq_tot, *self.original_shape[3:]))
        self._has_compressed_q_dimension = True
        self.update_original_shape()
        return self

    def decompress_q_dimension(self):
        """
        Converts the object from (q,...) to (qx,qy,qz,...) in-place, where len(q) = qx*qy*qz.
        """
        if not self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((*self.nq, *self.current_shape[1:]))
        self._has_compressed_q_dimension = False
        self.update_original_shape()
        return self

    def reduce_q(self, q_list: np.ndarray):
        """
        Reduces the object to the given list of momenta and returns a copy.
        Returns the object with compressed momentum dimension.
        """
        copy = deepcopy(self)

        if copy.has_compressed_q_dimension:
            copy.decompress_q_dimension()

        indices = np.indices(copy.current_shape[:3])
        mask = np.zeros(copy.current_shape[:3], dtype=bool)
        mask |= np.any(np.all(indices == np.array(q_list)[:, :, None, None, None], axis=1), axis=0)
        copy.mat = copy.mat[mask]

        copy.update_original_shape()
        copy._has_compressed_q_dimension = True
        return copy

    def find_q(self, q: tuple[int, int, int] = (0, 0, 0)):
        """
        Finds the matrix element for a given momentum q.
        """
        result = deepcopy(self).reduce_q(np.array(list(q))[None, ...])
        result._nq = q
        return result

    def map_to_full_bz(self, inverse_map: np.ndarray):
        """
        Maps the object to the full Brillouin zone using the inverse of the irreducible k-point mesh in-place and
        returns the original object.
        """
        if not self.has_compressed_q_dimension:
            raise ValueError("Mapping to full Brillouin zone only possible for compressed momentum dimension.")

        self.mat = self.mat[inverse_map, ...].reshape((np.prod(self.nq), *self.original_shape[1:]))
        self.update_original_shape()
        return self

    def _align_q_dimensions_for_operations(self, other: "IAmNonLocal"):
        """
        Adapts the frequency dimensions of two non-local objects to fit each other for addition or multiplication.
        """
        if not self.has_compressed_q_dimension and other.has_compressed_q_dimension:
            self.compress_q_dimension()
        if not other.has_compressed_q_dimension and self.has_compressed_q_dimension:
            other = other.compress_q_dimension()
        return other
