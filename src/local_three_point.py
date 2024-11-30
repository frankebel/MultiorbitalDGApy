import numpy as np

from local_n_point import LocalNPoint


class LocalThreePoint(LocalNPoint):
    def __init__(self, mat: np.ndarray, channel: str, full_niw_range: bool = True, full_niv_range: bool = True):
        super().__init__(mat, 4, 2, full_niv_range)
        self._channel = channel
        self._full_niw_range = full_niw_range

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def niw(self) -> int:
        return self.original_shape[-2] // 2 if self.full_niv_range else self.original_shape[-2]

    @property
    def full_niw_range(self) -> bool:
        return self._full_niw_range

    def cut_niw(self, niw_cut: int) -> "LocalThreePoint":
        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :]
        else:
            self.mat = self.mat[..., :niw_cut, :]

        self.original_shape = self.mat.shape
        return self

    def cut_niv(self, niv_cut: int) -> "LocalThreePoint":
        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niv - niv_cut : self.niv + niv_cut]
        else:
            self.mat = self.mat[..., :niv_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niw_and_niv(self, niw_cut: int, niv_cut: int) -> "LocalThreePoint":
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def invert(self) -> "LocalThreePoint":
        copy = self
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def to_compound_indices(self) -> "LocalThreePoint":
        if len(self.current_shape) == 3:
            return self
        # for compound indices, we have to add another fermionic frequency dimension v, which should be removed later
        self.mat = np.ascontiguousarray(
            np.einsum("...i,ij->...ij", self.mat, np.eye(2 * self.niv))
            .transpose(4, 0, 1, 5, 2, 3, 6)
            .reshape(2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv)
        )
        return self

    def to_full_indices(self, shape: tuple = None) -> "LocalThreePoint":
        if len(self.current_shape) == 6:  # [o1,o2,o3,o4,w,v]
            return self
        elif len(self.current_shape) == 3:  # [w,x1,x2]
            self.original_shape = shape if shape is not None else self.original_shape
            self.mat = np.ascontiguousarray(
                self.mat.reshape(
                    2 * self.niw + 1, self.n_bands, self.n_bands, 2 * self.niv, self.n_bands, self.n_bands, 2 * self.niv
                )
                .transpose(1, 2, 4, 5, 0, 3, 6)
                .diagonal(axis1=-2, axis2=-1)
            )
            return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")
