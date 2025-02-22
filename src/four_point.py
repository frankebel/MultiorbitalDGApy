from local_four_point import LocalFourPoint
from n_point_base import *


class FourPoint(LocalFourPoint, IAmNonLocal):
    """
    This class is used to represent a local four-point object in a given channel with a given number of momentum, orbital, bosonic and
    fermionic frequency dimensions that are be added to keep track of (re-)shaping.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: Channel = Channel.NONE,
        nq: tuple[int, int, int] = (1, 1, 1),
        nk: tuple[int, int, int] = (1, 1, 1),
        num_q_dimensions: int = 1,
        num_k_dimensions: int = 2,
        num_bosonic_frequency_dimensions: int = 1,
        num_fermionic_frequency_dimensions: int = 2,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        LocalFourPoint.__init__(
            self,
            mat,
            channel,
            num_bosonic_frequency_dimensions,
            num_fermionic_frequency_dimensions,
            full_niw_range,
            full_niv_range,
        )
        IAmNonLocal.__init__(self, nq, nk, num_q_dimensions, num_k_dimensions)

    @property
    def n_bands(self) -> int:
        return self.original_shape[self.num_q_dimensions + self.num_k_dimensions]

    def to_compound_indices(self) -> "FourPoint":
        r"""
        Converts the indices of the FourPoint object

        .. math:: F^{qkk'}_{lmm'l'}

        to compound indices

        .. math:: F^{q}_{c_1, c_2}.

        for a couple of shape cases. We group {k, l, m} and {k',m',l'} into two indices.

        .. math:: c_1 \;and\; c_2.

        k, k' and q here are four-component indices

        .. math:: k = \{\vec k, \nu\}.
        """
        if len(self.current_shape) == 4:  # [q, w, x1, x2]
            return self

        if self.num_bosonic_frequency_dimensions == 0 or self.num_q_dimensions == 0:
            raise ValueError("Cannot convert to compound indices if there is no q or w dimension.")
        if self.num_q_dimensions != 1 or self.num_bosonic_frequency_dimensions != 1:
            raise ValueError("Converting to compound indices only supported for one q and one w dimension.")

        self.original_shape = self.current_shape

        if self.num_fermionic_frequency_dimensions == 1:  # [q, o1, o2, o3, o4, w, v]
            self.extend_vn_dimension()

        if self.num_k_dimensions == 1:  # [q, k, o1, o2, o3, o4, w, ...]
            self.extend_to_two_dimensional_k_space()

        if self.num_k_dimensions == 0:  # [q, o1, o2, o3, o4, ...]
            if self.num_fermionic_frequency_dimensions == 0:  # [q, o1, o2, o3, o4, w]
                self.mat = self.mat.transpose(0, 5, 1, 2, 3, 4).reshape(
                    self.nq_tot, 2 * self.niw + 1, self.n_bands**2, self.n_bands**2
                )  # reshaping to [q,w,o1,o2,o3,o4] and then collecting {o1,o2} and {o3,o4} into two indices
                return self

            # [q, o1, o2, o3, o4, w, v, vp]
            self.mat = self.mat.transpose(0, 5, 1, 2, 6, 3, 4, 7).reshape(
                self.nq_tot, 2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
            )  # reshaping to [q,w,o1,o2,v,o3,o4,vp] and then collecting {o1,o2,v} and {o3,o4,vp} into two indices
            return self

        if self.num_fermionic_frequency_dimensions == 0:  # [q, k, k', o1, o2, o3, o4, w]
            self.mat = self.mat.transpose(0, 7, 1, 3, 4, 2, 5, 6).reshape(
                self.nq_tot, 2 * self.niw + 1, self.n_bands**2 * self.nk_tot, self.n_bands**2 * self.nk_tot
            )  # reshaping to [q,w,k,o1,o2,k',o3,o4] and then collecting {k,o1,o2} and {k',o3,o4} into two indices
            return self

        # [q, k, k', o1, o2, o3, o4, w, v, vp]
        self.mat = self.mat.transpose(0, 7, 1, 3, 4, 8, 2, 5, 6, 9).reshape(
            self.nq_tot,
            2 * self.niw + 1,
            self.n_bands**2 * self.nk_tot * 2 * self.niv,
            self.n_bands**2 * self.nk_tot * 2 * self.niv,
        )  # reshaping to [q,w,k,o1,o2,v,k',o3,o4,vp] and then collecting {k,o1,o2,v} and {k',o3,o4,vp} into two indices

        return self

    def to_full_indices(self, shape: tuple = None) -> "FourPoint":
        """
        Converts an object stored with compound indices to an object that has unraveled momentum, orbital and frequency axes.
        """
        if (
            len(self.current_shape)
            == self.num_q_dimensions
            + self.num_k_dimensions
            + self.num_orbital_dimensions
            + self.num_bosonic_frequency_dimensions
            + self.num_fermionic_frequency_dimensions
        ):
            return self

        if len(self.current_shape) != 4:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")

        self.original_shape = shape if shape is not None else self.original_shape
        # TODO: implement
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

        prefix = ":" * (self.num_q_dimensions + self.num_k_dimensions)
        permutation = f"{prefix}{split[0]}...->{prefix}{split[1]}..."

        copy = deepcopy(self)
        copy.mat = np.einsum(permutation, self.mat)
        return copy

    def extend_to_two_dimensional_k_space(self) -> "FourPoint":
        r"""
        Extends an object from one-dimensional in momentum k space to two-dimensional in momentum k space.

        .. math:: F^{\vec q \vec k} \to F^{\vec q \vec k \vec k}.
        """
        if self.num_k_dimensions == 0:
            raise ValueError("Cannot extend to two-dimensional k space if there are none.")
        if self.num_k_dimensions == 2:
            return self

        self.mat = np.einsum(":k...,ij->:kp...", self.mat, np.eye(self.nk_tot))
        self._num_k_dimensions = 2
        self.original_shape = self.current_shape
        return self

    def reduce_to_one_dimensional_k_space(self) -> "FourPoint":
        r"""
        Reduces an object to one-dimensional in momentum k space to two-dimensional k space by taking the diagonal in the k dimensions.

        .. math:: F^{\vec q \vec k \vec k} \to F^{\vec q \vec k}.
        """
        if self.num_k_dimensions != 2:
            raise ValueError("Cannot reduce to one-dimensional k space if there are not two.")

        self.mat = np.diagonal(self.mat, axis1=self.num_q_dimensions, axis2=self.num_q_dimensions + 1)
        self._num_k_dimensions = 1
        self.original_shape = self.current_shape
        return self
