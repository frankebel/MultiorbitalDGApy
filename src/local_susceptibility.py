import numpy as np

import config
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from matsubara_frequency_helper import FrequencyShift
from matsubara_frequency_helper import MFHelper
from i_have_channel import Channel


class LocalSusceptibility(LocalFourPoint):
    @staticmethod
    def create_generalized_chi(g2: LocalFourPoint, g_loc: LocalGreensFunction) -> "LocalSusceptibility":
        """gchi_r:1/eV^3 = beta:1/eV * (G2_r:1/eV^2 - 2 * GG:1/eV^2 delta_dens delta_w0)"""
        chi_mat = config.beta * g2.mat

        if g2.channel == Channel.DENS:
            ggv_mat = LocalSusceptibility._get_ggv_mat(g_loc, niv_slice=g2.niv)
            wn = MFHelper.get_wn_int(g2.niw)
            chi_mat[:, :, :, :, wn == 0] = config.beta * (g2.mat[:, :, :, :, wn == 0] - 2.0 * ggv_mat)

        return LocalSusceptibility(chi_mat, g2.channel, 1, 2, g2.full_niw_range, g2.full_niv_range)

    @staticmethod
    def create_generalized_chi0(
        g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
    ) -> "LocalSusceptibility":
        gchi0_mat = np.empty((g_loc.n_bands,) * 4 + (2 * config.niw + 1, 2 * config.niv), dtype=np.complex64)
        eye_bands = np.eye(g_loc.n_bands)

        wn = MFHelper.get_wn_int(config.niw)
        for index, current_wn in enumerate(wn):
            iws, iws2 = MFHelper.get_frequency_shift(current_wn, frequency_shift)

            g_left_mat = (
                g_loc.mat[..., g_loc.niv - config.niv + iws : g_loc.niv + config.niv + iws][:, None, None, :, :]
                * eye_bands[None, :, :, None, None]
            )
            g_right_mat = (
                np.swapaxes(g_loc.mat, 0, 1)[..., g_loc.niv - config.niv + iws2 : g_loc.niv + config.niv + iws2][
                    None, :, :, None, :
                ]
                * eye_bands[:, None, None, :, None]
            )

            gchi0_mat[..., index, :] = -config.beta * g_left_mat * g_right_mat

        return LocalSusceptibility(gchi0_mat, Channel.NONE, 1, 1, full_niw_range=True, full_niv_range=True)

    @staticmethod
    def _get_ggv_mat(g_loc: LocalGreensFunction, niv_slice: int = -1) -> np.ndarray:
        if niv_slice == -1:
            niv_slice = g_loc.niv
        g_loc_slice_mat = g_loc.mat[..., g_loc.niv - niv_slice : g_loc.niv + niv_slice]
        eye_bands = np.eye(g_loc.n_bands)
        g_left_mat = g_loc_slice_mat[:, None, None, :, :, None] * eye_bands[None, :, :, None, None, None]
        g_right_mat = (
            np.swapaxes(g_loc_slice_mat, 0, 1)[None, :, :, None, None, :] * eye_bands[None, None, None, :, :, None]
        )
        return g_left_mat * g_right_mat
