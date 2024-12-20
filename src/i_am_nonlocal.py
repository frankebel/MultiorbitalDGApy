import numpy as np


class IAmNonLocal:
    """
    Interface for objects that are momentum dependent.
    """

    def __init__(
        self, nq: tuple[int, int, int], nk: tuple[int, int, int], num_q_dimensions: int, num_k_dimensions: int
    ):
        self._nq = nq
        self._nk = nk

        assert num_q_dimensions == 1, "Only 1 q momentum dimension is supported."
        self._num_q_dimensions = num_q_dimensions

        assert num_k_dimensions in (0, 1, 2), "0 - 2 k momentum dimensions are supported."
        self._num_k_dimensions = num_k_dimensions

    @property
    def nq(self) -> tuple[int, int, int]:
        return self._nq

    @property
    def nq_tot(self) -> int:
        return np.prod(self.nq).astype(int)

    @property
    def nk(self) -> tuple[int, int, int]:
        return self._nk

    @property
    def nk_tot(self) -> int:
        return np.prod(self.nk).astype(int)

    @property
    def num_q_dimensions(self) -> int:
        return self._num_q_dimensions

    @property
    def num_k_dimensions(self) -> int:
        return self._num_k_dimensions
