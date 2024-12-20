import numpy as np
from enum import Enum
from multimethod import multimethod


class FrequencyShift(Enum):
    MINUS: str = "minus"
    PLUS: str = "plus"
    CENTER: str = "center"
    NONE: str = "none"


class MFHelper:
    @multimethod
    @staticmethod
    def wn(niw: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        """
        Returns integer numbers from [-niw, niw]. Additionally, a shift can be applied. If return_only_positive is
        set to True, only positive numbers [0, niw] are returned. If beta is provided, returns the real bosonic frequencies without an imaginary unit.
        """
        if return_only_positive:
            return np.arange(shift, niw + shift + 1)
        return np.arange(-niw + shift, niw + shift + 1)

    @multimethod
    @staticmethod
    def wn(niw: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        """
        Returns (real) bosonic matsubara frequencies from [-(2 niw)pi/beta, +(2 niw)pi/beta].
        """
        return np.pi / beta * 2 * MFHelper.wn(niw, shift, return_only_positive)

    @multimethod
    @staticmethod
    def vn(niv: int, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        """
        Returns integer numbers from [-niv, niv). Additionally, a shift can be applied. If return_only_positive is
        set to True, only positive numbers [0, niv) are returned. If beta is provided, returns the real fermionic frequencies without an imaginary unit.
        """
        if return_only_positive:
            return np.arange(shift, niv + shift)
        return np.arange(-niv + shift, niv + shift)

    @multimethod
    @staticmethod
    def vn(niv: int, beta: float, shift: int = 0, return_only_positive: bool = False) -> np.ndarray:
        """
        Returns (real) fermionic matsubara frequencies from [(-2 niv+1)pi/beta,  (+2 niv+1)pi/beta].
        """
        return np.pi / beta * (2 * MFHelper.vn(niv, shift, return_only_positive) + 1)

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
    def wn_slices_gen(mat: np.ndarray, niv_cut: int, niw: int) -> np.ndarray:
        niv = mat.shape[-1] // 2
        w = MFHelper.wn(niw)
        return np.moveaxis(np.array([mat[..., niv - niv_cut - iwn : niv + niv_cut - iwn] for iwn in w]), 0, -2)
