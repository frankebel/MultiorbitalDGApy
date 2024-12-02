import numpy as np

from local_n_point import LocalNPoint


class LocalTwoPoint(LocalNPoint):
    def __init__(
        self,
        mat: np.ndarray,
        num_bosonic_frequency_dimensions: int = 0,
        num_fermionic_frequency_dimensions: int = 1,
        full_niv_range: bool = True,
    ):
        super().__init__(
            mat, 2, num_bosonic_frequency_dimensions, num_fermionic_frequency_dimensions, True, full_niv_range
        )

    def invert(self) -> "LocalTwoPoint":
        copy = self
        copy.mat = np.linalg.inv(copy.mat.transpose(2, 0, 1)).transpose(1, 2, 0)
        return copy

    def to_compound_indices(self) -> "LocalTwoPoint":
        if len(self.current_shape) == 1:
            return self
        elif (
            len(self.current_shape)
            == self._num_orbital_dimensions
            + self._num_bosonic_frequency_dimensions
            + self._num_fermionic_frequency_dimensions
        ):
            self.mat = self.mat.reshape(np.prod(self.original_shape))
            return self
        else:
            raise ValueError(f"Converting to compound indices with shape {self.current_shape} not supported.")

    def to_full_indices(self, shape: tuple = None) -> "LocalTwoPoint":
        if len(self.current_shape) == 3:
            return self
        elif len(self.current_shape) == 1:
            self.original_shape = shape if shape is not None else self.original_shape
            self.mat: np.ndarray = self.mat.reshape(self.original_shape)
            return self
        else:
            raise ValueError(f"Converting to full indices with shape {self.current_shape} not supported.")
