import numpy as np


class MFHelper:
    @staticmethod
    def get_wn_int(niw: int, shift: int = 0, only_positive: bool = False) -> np.ndarray:
        if only_positive:
            return np.arange(shift, niw + shift + 1)
        return np.arange(-niw + shift, niw + shift + 1)

    @staticmethod
    def get_iwn(niw: int, beta: float, shift: int = 0, only_positive: bool = False) -> np.ndarray:
        return 1j * np.pi / beta * 2 * MFHelper.get_wn_int(niw, shift, only_positive)

    @staticmethod
    def get_vn_int(niv: int, shift: int = 0, only_positive: bool = False) -> np.ndarray:
        if only_positive:
            return np.arange(shift, niv + shift)
        return np.arange(-niv + shift, niv + shift)

    @staticmethod
    def get_ivn(niv: int, beta: float, shift: int = 0, only_positive: bool = False) -> np.ndarray:
        return 1j * np.pi / beta * (2 * MFHelper.get_vn_int(niv, shift, only_positive) + 1)
