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
        self._mat = mat.astype(np.complex64)
        self._original_shape = self.mat.shape

    @property
    def mat(self) -> np.ndarray:
        """
        Returns the underlying matrix.
        """
        return self._mat

    @mat.setter
    def mat(self, value: np.ndarray) -> None:
        self._mat = value

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
        return self.mat.nbytes * 1e-9

    def to_real(self) -> "IHaveMat":
        """
        Converts the matrix to real numbers. Returns it as complex type.
        """
        self.mat = self.mat.real.astype(np.complex64)
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

    def save(self, output_dir: str = "./", name: str = "please_give_me_a_name") -> None:
        """
        Saves the content of the matrix to a file.
        """
        np.save(os.path.join(output_dir, f"{name}.npy"), self.mat, allow_pickle=True)


class IAmNonLocal(IHaveMat, ABC):
    """
    Abstract interface for objects that are momentum dependent. Since we focus on ladder objects, we do not
    need more than one momentum variable for one- and two-particle quantities.
    """

    def __init__(self, mat: np.ndarray, nq: tuple[int, int, int], has_compressed_momentum_dimension: bool = False):
        super().__init__(mat)
        self._nq = nq
        self._has_compressed_q_dimension = has_compressed_momentum_dimension

    @property
    def nq(self) -> tuple[int, int, int]:
        """
        Returns the number of momenta in each direction.
        """
        return self._nq

    @property
    def nq_tot(self) -> int:
        """
        Returns the total number of momenta.
        """
        return np.prod(self.nq).astype(int)

    @property
    def has_compressed_q_dimension(self) -> bool:
        """
        Returns whether the underlying matrix has a compressed momentum dimension (q,...) or not (qx,qy,qz,...).
        """
        return self._has_compressed_q_dimension

    def shift_k_by_q(self, index: tuple | list[int] = (0, 0, 0)):
        """
        Shifts the momentum by the given value if the object does not have compressed momentum dimensions, i.e.,
        we require (qx,qy,qz,...).
        """
        if self.has_compressed_q_dimension:
            raise ValueError("Cannot shift momenta if the object has a compressed momentum dimension.")

        copy = deepcopy(self)
        copy.mat = np.roll(copy.mat, index, axis=(0, 1, 2))
        return copy

    def compress_q_dimension(self):
        """
        Converts the object from (qx,qy,qz,...) to (q,...), where len(q) = qx*qy*qz
        """
        if self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((self.nq_tot, *self.current_shape[3:]))
        self._has_compressed_q_dimension = True
        self.original_shape = self.current_shape
        return self

    def extend_q_dimension(self):
        """
        Converts the object from (q,...) to (qx,qy,qz,...), where len(q) = qx*qy*qz
        """
        if not self.has_compressed_q_dimension:
            return self

        self.mat = self.mat.reshape((*self.nq, *self.current_shape[1:]))
        self._has_compressed_q_dimension = False
        self.original_shape = self.current_shape
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
