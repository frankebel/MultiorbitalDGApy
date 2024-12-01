import numpy as np

from local_n_point import LocalNPoint, Channel


class LocalThreePoint(LocalNPoint):
    def __init__(
        self,
        mat: np.ndarray,
        channel: Channel = Channel.NONE,
        num_bosonic_frequency_dimensions: int = 1,
        num_fermionic_frequency_dimensions: int = 1,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        super().__init__(
            mat, 4, num_bosonic_frequency_dimensions, num_fermionic_frequency_dimensions, full_niw_range, full_niv_range
        )
        self._channel = channel

    @property
    def channel(self) -> Channel:
        return self._channel

    def invert(self) -> "LocalThreePoint":
        copy = self
        copy = copy.to_compound_indices()
        copy.mat = np.linalg.inv(copy.mat)
        return copy.to_full_indices()

    def to_compound_indices(self) -> "LocalThreePoint":
        if len(self.current_shape) == 3:
            return self
        # for compound indices, we have to add another fermionic frequency dimension v, which needs to be removed later
        self.mat = (
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
            self.mat = (
                self.mat.reshape(
                    2 * self.niw + 1, self.n_bands, self.n_bands, 2 * self.niv, self.n_bands, self.n_bands, 2 * self.niv
                )
                .transpose(1, 2, 4, 5, 0, 3, 6)
                .diagonal(axis1=-2, axis2=-1)
            )

            return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")
