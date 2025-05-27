from copy import deepcopy

import numpy as np

from scdga.interaction import Interaction, LocalInteraction
from scdga.local_four_point import LocalFourPoint
from scdga.n_point_base import IAmNonLocal, SpinChannel, FrequencyNotation


class FourPoint(LocalFourPoint, IAmNonLocal):
    """
    This class is used to represent a non-local four-point object in a given channel with a given number of momentum,
    orbital, bosonic and fermionic frequency dimensions that are be added to keep track of (re-)shaping.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ):
        LocalFourPoint.__init__(
            self,
            mat,
            channel,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            frequency_notation,
        )
        IAmNonLocal.__init__(self, mat, nq, has_compressed_q_dimension)

    @property
    def n_bands(self) -> int:
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def __add__(self, other) -> "FourPoint":
        """
        Addition for FourPoint objects. Allows for A + B = C.
        """
        return self.add(other)

    def __radd__(self, other) -> "FourPoint":
        """
        Addition for FourPoint objects. Allows for A + B = C.
        """
        return self.add(other)

    def __sub__(self, other) -> "FourPoint":
        """
        Subtraction for FourPoint objects. Allows for A + B = C.
        """
        return self.sub(other)

    def __rsub__(self, other) -> "FourPoint":
        """
        Subtraction for FourPoint objects. Allows for A + B = C.
        """
        return self.sub(other)

    def __mul__(self, other) -> "FourPoint":
        r"""
        Allows for the multiplication with a number, a numpy array or a FourPoint object. In the latter case,
        we require both objects to only have one niv dimension, such that

        .. math:: A_{abcd}^{qv} * B_{dcef}^{qv'} = C_{abef}^{qvv'}.
        Returns the object in the half niw range.
        """
        return self.mul(other)

    def __matmul__(self, other) -> "FourPoint":
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=True)

    def __rmatmul__(self, other) -> "FourPoint":
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, left_hand_side=False)

    def __pow__(self, power, modulo=None) -> "FourPoint":
        """
        Exponentiation for FourPoint objects. Allows for A ** n = B, where n is an integer. If n < 0, then we
        exponentiate the inverse of A |n| times, i.e., A ** (-3) = A^(-1) ** 3.
        """
        return self.pow(power, FourPoint.identity_like(self))

    def sum_over_vn(self, beta: float, axis: tuple = (-1,)) -> "FourPoint":
        """
        Sums over specific fermionic frequency dimensions and multiplies with the correct prefactor 1/beta^(n_dim).
        """
        if len(axis) > self.num_vn_dimensions:
            raise ValueError(f"Cannot sum over more fermionic axes than available in {self.current_shape}.")
        copy_mat = 1 / beta ** len(axis) * np.sum(self.mat, axis=axis)
        self.update_original_shape()
        return FourPoint(
            copy_mat,
            self.channel,
            self.nq,
            self.num_wn_dimensions,
            self.num_vn_dimensions - len(axis),
            self.full_niw_range,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        )

    def sum_over_orbitals(self, orbital_contraction: str = "abcd->ad") -> "FourPoint":
        """
        Sums over the given orbitals.
        """
        split = orbital_contraction.split("->")
        if len(split[0]) != 4 or len(split[1]) > len(split[0]):
            raise ValueError("Invalid orbital contraction.")

        permutation = (
            f"i{split[0]}...->i{split[1]}..."
            if self.has_compressed_q_dimension
            else f"ijk{split[0]}...->ijk{split[1]}..."
        )

        self.mat = np.einsum(permutation, self.mat)
        diff = len(split[0]) - len(split[1])
        self.update_original_shape()
        self._num_orbital_dimensions -= diff
        return self

    def to_compound_indices(self) -> "FourPoint":
        r"""
        Converts the indices of the FourPoint object

        .. math:: F^{qvv'}_{lmm'l'}
        to compound indices

        .. math:: F^{q}_{c_1, c_2}.
        Always returns the object with a compressed momentum dimension.
        """
        if len(self.current_shape) == 4:  # [q, w, x1, x2]
            return self

        if not self.has_compressed_q_dimension:
            self.compress_q_dimension()

        self.update_original_shape()

        if (
            self.num_wn_dimensions == 0  # [q, o1, o2, o3, o4, v, vp]
        ):  # we assume that the object has 2 fermionic frequency dimensions, as is the case for pairing objects.
            if self.num_vn_dimensions != 2:
                raise ValueError(
                    "Object must have 2 fermionic frequency dimensions if it does not have any w dimension."
                )
            self.mat = self.mat.reshape(self.nq_tot, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv)
            return self

        if self.num_vn_dimensions == 0:  # [q, o1, o2, o3, o4, w]
            self.mat = self.mat.transpose(0, 5, 1, 2, 3, 4).reshape(
                self.nq_tot, 2 * self.niw + 1, self.n_bands**2, self.n_bands**2
            )  # reshaping to [q,w,o1,o2,o3,o4] and then collecting {o1,o2} and {o3,o4} into two indices
            return self

        if self.num_vn_dimensions == 1:  # [q, o1, o2, o3, o4, w, v]
            self.extend_vn_to_diagonal()

        # [q, o1, o2, o3, o4, w, v, vp]
        self.mat = self.mat.transpose(0, 5, 1, 2, 6, 3, 4, 7).reshape(
            self.nq_tot, 2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )  # reshaping to [q,w,o1,o2,v,o3,o4,vp] and then collecting {o1,o2,v} and {o3,o4,vp} into two indices

        return self

    def to_full_indices(self, shape: tuple = None) -> "FourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled momentum,
        orbital and frequency axes. Always returns the object with a compressed momentum dimension.
        """
        if (
            len(self.current_shape) == 1 + self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions
            and self.has_compressed_q_dimension
        ):
            return self
        elif (
            len(self.current_shape) == 3 + self.num_orbital_dimensions + self.num_wn_dimensions + self.num_vn_dimensions
            and not self.has_compressed_q_dimension
        ):
            return self

        if (len(self.current_shape) != 4 and self.has_compressed_q_dimension) or (
            len(self.current_shape) != 6 and not self.has_compressed_q_dimension
        ):  # (q,w,x1,x2) or (qx,qy,qz,w,x1,x2)
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

        if self.num_wn_dimensions != 1:
            raise ValueError("Number of bosonic frequency dimensions must be 1.")

        self.original_shape = shape if shape is not None else self.original_shape

        if self.num_vn_dimensions == 0:  # original was [q,o1,o2,o3,o4,w]
            self.mat = self.mat.reshape(
                (self.nq_tot,) + (2 * self.niw + 1,) + (self.n_bands,) * self.num_orbital_dimensions
            ).transpose(0, 2, 3, 4, 5, 1)
            self._has_compressed_momentum_dimension = True
            return self

        compound_index_shape = (self.n_bands, self.n_bands, 2 * self.niv)

        self.mat = self.mat.reshape((self.nq_tot,) + (2 * self.niw + 1,) + compound_index_shape * 2).transpose(
            0, 2, 3, 5, 6, 1, 4, 7
        )

        if self.num_vn_dimensions == 1:  # original was [q,o1,o2,o3,o4,w,v]
            self.mat = self.mat.diagonal(axis1=-2, axis2=-1)
        return self

    def permute_orbitals(self, permutation: str = "abcd->abcd") -> "FourPoint":
        """
        Permutes the orbitals of the four-point object.
        """
        split = permutation.split("->")
        if len(split) != 2 or len(split[0]) != 4 or len(split[1]) != 4:
            raise ValueError("Invalid permutation.")

        if split[0] == split[1]:
            return self

        permutation = (
            f"i{split[0]}...->i{split[1]}..."
            if self.has_compressed_q_dimension
            else f"ijk{split[0]}...->ijk{split[1]}..."
        )

        copy = deepcopy(self)
        copy.mat = np.einsum(permutation, copy.mat, optimize=True)
        return copy

    def add(self, other) -> "FourPoint":
        """
        Helper method that allows for addition of (non-)local FourPoint objects.
        Depending on the number of frequency and momentum dimensions, the objects have to be added differently.
        """
        if not isinstance(
            other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction, np.ndarray, float, int, complex)
        ):
            raise ValueError(f"Operations '+/-' for {type(self)} and {type(other)} not supported.")

        if isinstance(other, (np.ndarray, float, int, complex)):
            return FourPoint(
                self.mat + other,
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if isinstance(other, (Interaction, LocalInteraction)):
            self.compress_q_dimension()

            other_mat = other.mat[None, ...] if not isinstance(other, Interaction) else other.mat
            other_mat = other_mat.reshape(other.mat.shape + (1,) * (self.num_wn_dimensions + self.num_vn_dimensions))
            return FourPoint(
                self.mat + other_mat,
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        self_full_niw_range = self.full_niw_range
        other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()

        if not isinstance(other, FourPoint):
            other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)
            result = FourPoint(
                (
                    self.mat
                    + (other.mat[None, ...] if self.has_compressed_q_dimension else other.mat[None, None, None, ...])
                ),
                channel,
                self.nq,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                False,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

            if self_full_niw_range:
                self.to_full_niw_range()
            if other_full_niw_range:
                other = other.to_full_niw_range()

            other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
            return result

        other = self._align_q_dimensions_for_operations(other)
        other, self_extended, other_extended = self._align_frequency_dimensions_for_operation(other)

        result = FourPoint(
            self.mat + other.mat,
            channel,
            self.nq,
            self.num_wn_dimensions,
            self.num_vn_dimensions,
            False,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        )

        if self_full_niw_range:
            self.to_full_niw_range()
        if other_full_niw_range:
            other = other.to_full_niw_range()

        other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
        return result

    def sub(self, other) -> "FourPoint":
        """
        Helper method that allows for in-place subtraction of (non-)local FourPoint objects.
        Depending on the number of frequency and momentum dimensions, the objects have to be subtracted differently.
        """
        return self.add(-other)

    def mul(self, other) -> "FourPoint":
        r"""
        Allows for the multiplication with a number, a numpy array or a FourPoint object. In the latter case,
        we require both objects to only have one niv dimension, such that

        .. math:: A_{abcd}^{qv} * B_{dcef}^{qv'} = C_{abef}^{qvv'}.
        Returns the object in the half niw range.
        """
        if not isinstance(other, (int, float, complex, np.ndarray, LocalFourPoint)):
            raise ValueError("Multiplication only supported with numbers, numpy arrays or FourPoint objects.")

        if not isinstance(other, LocalFourPoint):
            copy = deepcopy(self)
            copy.mat *= other
            return copy

        if self.num_vn_dimensions != 1 or other.num_vn_dimensions != 1:
            raise ValueError("Both objects must have only one fermionic frequency dimension.")

        is_self_full_niw_range = self.full_niw_range
        is_other_full_niw_range = other.full_niw_range

        self.to_half_niw_range()
        other = other.to_half_niw_range()
        result_mat = self.times("qabcdwv,qdcefwp->qabefwvp", other)

        if is_self_full_niw_range:
            self.to_full_niw_range()
        if is_other_full_niw_range:
            other = other.to_full_niw_range()

        return FourPoint(result_mat, self.channel, self.nq, 1, 2, False, True, True, self.frequency_notation)

    def matmul(self, other, left_hand_side: bool = True) -> "FourPoint":
        """
        Helper method that allows for matrix multiplication for (non-)local FourPoint objects. Depending on the
        number of frequency and momentum dimensions, the objects have to be multiplied differently. If the objects
        come in with only half of their niw range, they will be returned in half of their niw range to save memory.
        """
        if not isinstance(other, (FourPoint, LocalFourPoint, Interaction, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalInteraction, Interaction)):
            is_local = not isinstance(other, Interaction)
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()

            einsum_str_map = {
                0: f"qabijw,{q_prefix}jief->qabefw" if left_hand_side else f"{q_prefix}abij,qjiefw->qabefw",
                1: f"qabijwv,{q_prefix}jief->qabefwv" if left_hand_side else f"{q_prefix}abij,qjiefwv->qabefwv",
                2: f"qabijwvp,{q_prefix}jief->qabefwvp" if left_hand_side else f"{q_prefix}abij,qjiefwvp->qabefwvp",
            }
            einsum_str = einsum_str_map.get(self.num_vn_dimensions)

            return FourPoint(
                (
                    np.einsum(einsum_str, self.mat, other.mat, optimize=True)
                    if left_hand_side
                    else np.einsum(einsum_str, other.mat, self.mat, optimize=True)
                ),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        is_local = not isinstance(other, FourPoint)
        channel = self.channel if self.channel != SpinChannel.NONE else other.channel

        if self.num_vn_dimensions == 0 or other.num_vn_dimensions == 0:
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()
            if not is_local:
                other = other.compress_q_dimension()

            # special case if one of two (Local)FourPoint objects has no fermionic frequency dimensions
            # straightforward contraction is saving memory as we do not have to add fermionic frequency dimensions
            einsum_str = {
                (0, 2): (
                    f"qabcdw,{q_prefix}dcefwvp->qabefwvp" if left_hand_side else f"{q_prefix}abcdwvp,dcefw->qabefwvp"
                ),
                (0, 1): f"qabcdw,{q_prefix}dcefwv->qabefwv" if left_hand_side else f"{q_prefix}abcdwv,dcefw->qabefwv",
                (0, 0): f"qabcdw,{q_prefix}dcefw->qabefw" if left_hand_side else f"{q_prefix}abcdw,dcefw->qabefw",
                (1, 0): f"qabcdwv,{q_prefix}dcefw->qabefwv" if left_hand_side else f"{q_prefix}abcdw,dcefwv->qabefwv",
                (2, 0): (
                    f"qabcdwvp,{q_prefix}dcefw->qabefwvp" if left_hand_side else f"{q_prefix}abcdw,dcefwvp->qabefwvp"
                ),
            }.get((self.num_vn_dimensions, other.num_vn_dimensions))

            return FourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
                channel,
                self.nq,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        if self.num_vn_dimensions == 1 or other.num_vn_dimensions == 1:
            einsum_str = {
                (1, 1): "qabcdwv,qdcefwv->qabefwv",
                (1, 2): "qabcdwv,qdcefwvp->qabefwvp" if left_hand_side else "qabcdwvp,qdcefwp->qabefwvp",
                (2, 1): "qabcdwvp,qdcefwp->qabefwvp" if left_hand_side else "qabcdwv,qdcefwvp->qabefwvp",
            }.get((self.num_vn_dimensions, other.num_vn_dimensions))
            new_mat = np.einsum(einsum_str, self.mat, other.mat, optimize=True)
            max_vn_dim = max(self.num_vn_dimensions, other.num_vn_dimensions)
            return FourPoint(
                new_mat, channel, self.nq, self.num_wn_dimensions, max_vn_dim, False, self.full_niv_range, True
            )

        is_self_full_niw_range = self.full_niw_range
        is_other_full_niw_range = other.full_niw_range

        self.to_half_niw_range().to_compound_indices()
        other = other.to_half_niw_range().to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = (
            np.matmul(self.mat, other.mat[None, ...] if is_local else other.mat)
            if left_hand_side
            else np.matmul(other.mat[None, ...] if is_local else other.mat, self.mat)
        )

        self.to_full_indices()
        if is_self_full_niw_range:
            self.to_full_niw_range()
        other = other.to_full_indices()
        if is_other_full_niw_range:
            other = other.to_full_niw_range()

        return FourPoint(
            new_mat,
            channel,
            self.nq,
            self.num_wn_dimensions,
            2,
            False,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        ).to_full_indices(self.original_shape)

    def rotate_orbitals(self, theta: float = np.pi):
        r"""
        Rotates the orbitals of the four-point object around the angle :math:`\theta`. :math:`\theta` must be given in
        radians and the number of orbitals needs to be 2.
        """
        if self.n_bands != 2:
            raise ValueError("Rotating the orbitals is only allowed for objects that have two bands.")

        copy = deepcopy(self)
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        einsum_str = (
            "pi,qj,rk,sl,xpqrs...->xijkl..."
            if self.has_compressed_q_dimension
            else "pi,qj,rk,sl,xyzpqrs...->xyzijkl..."
        )
        copy.mat = np.einsum(einsum_str, np.conj(r.T), np.conj(r.T), r, r, copy.mat, optimize=True)
        return copy

    @staticmethod
    def load(
        filename: str,
        channel: SpinChannel = SpinChannel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_wn_dimensions: int = 1,
        num_vn_dimensions: int = 2,
        full_niw_range: bool = False,
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = True,
        frequency_notation: FrequencyNotation = FrequencyNotation.PH,
    ) -> "FourPoint":
        """
        Loads a FourPoint object from a file. The file must be of type '.npy'.
        """
        return FourPoint(
            np.load(filename, allow_pickle=False),
            channel,
            nq,
            num_wn_dimensions,
            num_vn_dimensions,
            full_niw_range,
            full_niv_range,
            has_compressed_q_dimension,
            frequency_notation,
        )

    @staticmethod
    def identity(
        n_bands: int,
        niw: int,
        niv: int,
        nq_tot: int = 1,
        nq: tuple[int, int, int] = (1, 1, 1),
        num_vn_dimensions: int = 2,
    ) -> "FourPoint":
        """
        Creates an identity (matrix in compound index notation is unity) for the FourPoint object.
        """
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        full_shape = (nq_tot,) + (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size)[None, None, ...], (nq_tot, 2 * niw + 1, 1, 1))

        result = FourPoint(
            mat, nq=nq, num_vn_dimensions=num_vn_dimensions, has_compressed_q_dimension=True
        ).to_full_indices(full_shape)
        if num_vn_dimensions == 1:
            return result.take_vn_diagonal()
        return result

    @staticmethod
    def identity_like(other: "FourPoint") -> "FourPoint":
        """
        Creates an identity (matrix in compound index notation is unity) for the FourPoint object from the shape
        of another FourPoint object.
        """
        return FourPoint.identity(other.n_bands, other.niw, other.niv, other.nq_tot, other.nq, other.num_vn_dimensions)
