import numpy as np

from local_n_point import LocalNPoint


class LocalFourPoint(LocalNPoint):
    def __init__(self, mat: np.ndarray, channel: str, full_niw_range: bool = True, full_niv_range: bool = True):
        super().__init__(mat, 4, 3, full_niv_range)
        self._channel = channel
        self._full_niw_range = full_niw_range

    @property
    def channel(self):
        return self._channel

    @property
    def niw(self):
        return self.original_shape[-3] // 2 if self.full_niv_range else self.original_shape[-3]

    @property
    def full_niw_range(self):
        return self._full_niw_range

    def cut_niw(self, niw_cut):
        if niw_cut > self.niw:
            raise ValueError("Cannot cut more bosonic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niw - niw_cut : self.niw + niw_cut + 1, :, :]
        else:
            self.mat = self.mat[..., :niw_cut, :, :]

        self.original_shape = self.mat.shape
        return self

    def cut_niv(self, niv_cut):
        if niv_cut > self.niv:
            raise ValueError("Cannot cut more fermionic frequencies than the object has.")

        if self.full_niv_range:
            self.mat = self.mat[..., self.niv - niv_cut : self.niv + niv_cut, self.niv - niv_cut : self.niv + niv_cut]
        else:
            self.mat = self.mat[..., :niv_cut, :niv_cut]

        self.original_shape = self.mat.shape
        return self

    def cut_niw_and_niv(self, niw_cut, niv_cut):
        return self.cut_niw(niw_cut).cut_niv(niv_cut)

    def symmetrize_v_vp(self):
        self.mat = 0.5 * (self.mat + np.swapaxes(self.mat, -1, -2))
        return self

    def invert(self):
        copy = self
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def to_compound_indices(self):
        self.original_shape = self.mat.shape
        self.mat = self.mat.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
            2 * self.niw + 1, self.n_bands**2 * 2 * self.niv, self.n_bands**2 * 2 * self.niv
        )
        return self

    def to_full_indices(self):
        if len(self.current_shape) == 7:  # [o1,o2,o3,o4,w,v,vp]
            return self
        elif len(self.current_shape) == 3:  # [w,x1,x2]
            self.mat = self.mat.reshape(
                2 * self.niw + 1, self.n_bands, self.n_bands, 2 * self.niv, self.n_bands, self.n_bands, 2 * self.niv
            ).transpose(1, 2, 4, 5, 0, 3, 6)
            return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")
