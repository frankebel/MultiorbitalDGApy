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
        """
        Returns w and v for the given frequency shift notation.
        """
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
    def get_frequencies_for_ph_to_pp_channel_conversion(
        niw: int, niv: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the new
        .. math:: (w', v_1', v_2')
        indices for the conversion of a ph to a pp channel.\n
        .. math::  F_{ph_bar}[...] = F_ph[w',v_1',v_2'] \n
        .. math::  (w,v_1,v_2) -> (w',v_1',v_2') = (v_1 + v_2 - w, v_1, v_2)
        """
        niw, niv = niw // 2, min(niw // 2, niv // 2)
        iw, iv, ivp = MFHelper._get_frequencies_for_channel_conversion(niw, niv)
        return niw + iv + ivp - iw, niv + iv, niv + ivp

    @staticmethod
    def get_frequencies_for_ph_to_ph_bar_channel_conversion(
        niw: int, niv: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the new
        .. math:: (w',v_1',v_2')
        indices for the conversion of a ph to a ph_bar channel.\n
        .. math::  F_ph_bar[...] = F_ph[w',v_1',v_2'] \n
        .. math::  (w,v_1,v_2) -> (w',v_1',v_2') = (v_1-v_2, v_1, v_1-w) \n
        """
        niw, niv = niw // 2, min(niw // 2, niv // 2)
        iw, iv, ivp = MFHelper._get_frequencies_for_channel_conversion(niw, niv)
        return niw + iv - ivp, niv + iv, niv + iv - iw

    @staticmethod
    def _get_frequencies_for_channel_conversion(niw: int, niv: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wn, vn = MFHelper.wn(niw), MFHelper.vn(niv)
        return wn[:, None, None], vn[None, :, None], vn[None, None, :]

    @staticmethod
    def wn_slices_gen(mat: np.ndarray, niv_cut: int, niw: int) -> np.ndarray:
        niv = mat.shape[-1] // 2
        w = MFHelper.wn(niw)
        return np.moveaxis(np.array([mat[..., niv - niv_cut - iwn : niv + niv_cut - iwn] for iwn in w]), 0, -2)

    @staticmethod
    def fermionic_full_nu_range(mat: np.ndarray, axis=(-1,)):
        """Build full fermionic object from positive frequencies only along axis."""
        return np.concatenate((np.conj(np.flip(mat, axis)), mat), axis=axis)
