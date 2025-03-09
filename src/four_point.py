import numpy as np

from local_four_point import LocalFourPoint
from n_point_base import *
from interaction import NonLocalInteraction, LocalInteraction


class FourPoint(LocalFourPoint, IAmNonLocal):
    """
    This class is used to represent a local four-point object in a given channel with a given number of momentum, orbital, bosonic and
    fermionic frequency dimensions that are be added to keep track of (re-)shaping.
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
        has_compressed_momentum_dimension: bool = False,
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
        IAmNonLocal.__init__(self, mat, nq, has_compressed_momentum_dimension)

    @property
    def n_bands(self) -> int:
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def __add__(self, other):
        """
        Addition for FourPoint objects. Allows for A + B = C.
        """
        return self.add(other)

    def __radd__(self, other):
        """
        Addition for FourPoint objects. Allows for A + B = C.
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction for FourPoint objects. Allows for A + B = C.
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """
        Subtraction for FourPoint objects. Allows for A + B = C.
        """
        return self.__add__(-other)

    def __matmul__(self, other):
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, True)

    def __rmatmul__(self, other):
        """
        Matrix multiplication for FourPoint objects. Allows for A @ B = C using compound indices.
        """
        return self.matmul(other, False)

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

        if self.num_wn_dimensions != 1:
            raise ValueError("Converting to compound indices only supported for one w dimension.")

        if not self.has_compressed_q_dimension:
            self.compress_q_dimension()

        self.original_shape = self.current_shape

        if self.num_vn_dimensions == 0:  # [q, o1, o2, o3, o4, w]
            self.mat = self.mat.transpose(0, 5, 1, 2, 3, 4).reshape(
                self.nq_tot, 2 * self.niw + 1, self.n_bands**2, self.n_bands**2
            )  # reshaping to [q,w,o1,o2,o3,o4] and then collecting {o1,o2} and {o3,o4} into two indices
            return self

        if self.num_vn_dimensions == 1:  # [q, o1, o2, o3, o4, w, v]
            self.extend_vn_dimension()

        # [q, o1, o2, o3, o4, w, v, vp]
        self.mat = self.mat.transpose(0, 5, 1, 2, 6, 3, 4, 7).reshape(
            self.nq_tot,
            2 * self.niw + 1,
            self.n_bands**2 * 2 * self.niv,
            self.n_bands**2 * 2 * self.niv,
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

        if len(self.current_shape) != 4 or len(self.current_shape) != 6:  # (q,w,x1,x2) or (qx,qy,qz,w,x1,x2)
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
        if (
            len(split) != 2
            or len(split[0]) != self.num_orbital_dimensions
            or len(split[1]) != self.num_orbital_dimensions
        ):
            raise ValueError("Invalid permutation.")

        permutation = (
            f":{split[0]}...->:{split[1]}..."
            if self.has_compressed_q_dimension
            else f":::{split[0]}...->:::{split[1]}..."
        )

        copy = deepcopy(self)
        copy.mat = np.einsum(permutation, self.mat, optimize=True)
        return copy

    def add(self, other) -> "FourPoint":
        """
        Helper method that allows for in-place addition and subtraction of (non-)local FourPoint objects.
        Depending on the number of frequency dimensions, the objects have to be added differently.
        """
        if not isinstance(other, (FourPoint, LocalFourPoint, np.ndarray, float, int, complex)):
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

        if isinstance(other, LocalFourPoint):
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
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

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
            self.full_niw_range,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        )

        other = self._revert_frequency_dimensions_after_operation(other, other_extended, self_extended)
        return result

    def matmul(self, other, left_hand_side: bool = True) -> "FourPoint":
        if not isinstance(other, (FourPoint, LocalFourPoint, NonLocalInteraction, LocalInteraction)):
            raise ValueError(f"Multiplication {type(self)} @ {type(other)} not supported.")

        if isinstance(other, (LocalInteraction, NonLocalInteraction)):
            is_local = isinstance(other, LocalInteraction)
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()
            if not is_local:
                other = other.compress_q_dimension()

            einsum_str_map = {
                0: f"qabijw,{q_prefix}jief->qabefw" if left_hand_side else f"{q_prefix}abij,qjiefw->qabefw",
                1: f"qabijwv,{q_prefix}jief->qabefwv" if left_hand_side else f"{q_prefix}abij,qjiefwv->qabefwv",
                2: f"qabijwvp,{q_prefix}jief->qabefwvp" if left_hand_side else f"{q_prefix}abij,qjiefwvp->qabefwvp",
            }
            einsum_str = einsum_str_map.get(self.num_vn_dimensions)

            return FourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                self.num_vn_dimensions,
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        is_local = isinstance(other, LocalFourPoint)
        if (
            self.num_wn_dimensions == 1
            and other.num_wn_dimensions == 1
            and (self.num_vn_dimensions == 0 or other.num_vn_dimensions == 0)
        ):
            q_prefix = "" if is_local else "q"

            self.compress_q_dimension()
            if not is_local:
                other = other.compress_q_dimension()

            # special case if one of two (Local)FourPoint object has no fermionic frequency dimensions
            # straightforward contraction is saving memory as we do not have to add fermionic frequency dimensions
            einsum_str = {
                (0, 2): f"qabcdw,{q_prefix}dcefwvp->qabefwvp",
                (0, 1): f"qabcdw,{q_prefix}dcefwv->qabefwv",
                (0, 0): f"qabcdw,{q_prefix}dcefw->qabefw",
                (1, 0): f"qabcdwv,{q_prefix}dcefw->qabefwv",
                (2, 0): f"qabcdwvp,{q_prefix}dcefw->qabefwvp",
            }.get((self.num_vn_dimensions, other.num_vn_dimensions))

            return FourPoint(
                np.einsum(einsum_str, self.mat, other.mat, optimize=True),
                self.channel,
                self.nq,
                self.num_wn_dimensions,
                max(self.num_vn_dimensions, other.num_vn_dimensions),
                self.full_niw_range,
                self.full_niv_range,
                self.has_compressed_q_dimension,
                self.frequency_notation,
            )

        self.to_compound_indices()
        other = other.to_compound_indices()
        # for __matmul__ self needs to be the LHS object, for __rmatmul__ self needs to be the RHS object
        new_mat = (
            np.matmul(self.mat, other.mat[None, ...] if is_local else other.mat)
            if left_hand_side
            else np.matmul(other.mat[None, ...] if is_local else other.mat, self.mat)
        )
        self.to_full_indices()
        other = other.to_full_indices()

        other_shape = (self.nq_tot, *other.original_shape) if is_local else other.original_shape

        return FourPoint(
            new_mat,
            self.channel,
            self.nq,
            self.num_wn_dimensions,
            max(self.num_vn_dimensions, other.num_vn_dimensions),
            self.full_niw_range,
            self.full_niv_range,
            self.has_compressed_q_dimension,
            self.frequency_notation,
        ).to_full_indices(self.original_shape if self.num_vn_dimensions == 2 else other_shape)

    @staticmethod
    def identity(
        n_bands: int, niw: int, niv: int, nq: tuple[int, int, int] = (1, 1, 1), num_vn_dimensions: int = 2
    ) -> "FourPoint":
        if num_vn_dimensions not in (1, 2):
            raise ValueError("Invalid number of fermionic frequency dimensions.")
        nq_tot = np.prod(nq)
        full_shape = (nq_tot,) + (n_bands,) * 4 + (2 * niw + 1,) + (2 * niv,) * num_vn_dimensions
        compound_index_size = 2 * niv * n_bands**2
        mat = np.tile(np.eye(compound_index_size)[None, None, ...], (nq_tot, 2 * niw + 1, 1, 1))

        result = FourPoint(mat, nq=nq, num_vn_dimensions=num_vn_dimensions).to_full_indices(full_shape)
        if num_vn_dimensions == 1:
            return result.compress_vn_dimensions()
        return result
