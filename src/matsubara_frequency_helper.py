import numpy as np
from enum import Enum


class FrequencyShift(Enum):
    MINUS: str = "minus"
    PLUS: str = "plus"
    CENTER: str = "center"
    NONE: str = "none"


class MFHelper:
    @staticmethod
    def get_wn_int(niw: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        if return_only_positive:
            return np.arange(shift, niw + shift + 1)
        return np.arange(-niw + shift, niw + shift + 1)

    @staticmethod
    def get_iwn(niw: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        return 1j * np.pi / beta * 2 * MFHelper.get_wn_int(niw, shift, return_only_positive)

    @staticmethod
    def get_vn_int(niv: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        if return_only_positive:
            return np.arange(shift, niv + shift)
        return np.arange(-niv + shift, niv + shift)

    @staticmethod
    def get_ivn(niv: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        return 1j * np.pi / beta * (2 * MFHelper.get_vn_int(niv, shift, return_only_positive) + 1)

    @staticmethod
    def get_frequency_shift(wn: int, freq_notation: FrequencyShift) -> (int, int):
        if freq_notation == FrequencyShift.PLUS:  # for something like chi_0[w,v] = -beta G(v) * G(v+w)
            return 0, wn
        elif freq_notation == FrequencyShift.MINUS:  # for something like chi_0[w,v] = -beta G(v) * G(v-w)
            return 0, -wn
        elif freq_notation == FrequencyShift.CENTER:  # for something like chi_0[w,v] = -beta G(v+w//2) * G(v-w//2-w%2)
            return wn // 2, -(wn // 2 + wn % 2)
        else:
            raise NotImplementedError(
                f"Frequency notation '{freq_notation}' is not in list {[s for s in FrequencyShift]}."
            )

    @staticmethod
    def extend_last_frequency_axis_to_diagonal(mat: np.ndarray) -> np.ndarray:
        return np.einsum("...i,ij->...ij", mat, np.eye(mat.shape[-1]))

    @staticmethod
    def compress_last_two_frequency_dimensions_to_single_dimension(mat: np.ndarray) -> np.ndarray:
        return mat.diagonal(axis1=-2, axis2=-1)
