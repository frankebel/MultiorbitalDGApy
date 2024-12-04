import numpy as np

import config
from i_have_channel import Channel
from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_three_point import LocalThreePoint
from matsubara_frequency_helper import MFHelper, FrequencyShift


class LocalSusceptibility(LocalFourPoint):
    @staticmethod
    def create_generalized_chi(g2: LocalFourPoint, g_loc: LocalGreensFunction) -> "LocalSusceptibility":
        """gchi_r:1/eV^3 = beta:1/eV * (G2_r:1/eV^2 - 2 * GG:1/eV^2 delta_dens delta_w0)"""
        chi = config.beta * g2

        if g2.channel == Channel.DENS:
            ggv_mat = LocalSusceptibility._get_ggv_mat(g_loc, niv_slice=g2.niv)
            wn = MFHelper.get_wn_int(g2.niw)
            chi[:, :, :, :, wn == 0] = config.beta * (g2[:, :, :, :, wn == 0] - 2.0 * ggv_mat)

        return LocalSusceptibility(
            chi.mat,
            chi.channel,
            full_niw_range=chi.full_niw_range,
            full_niv_range=chi.full_niv_range,
        )

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

        gchi0_mat = MFHelper.extend_last_frequency_axis_to_diagonal(gchi0_mat)
        return LocalSusceptibility(gchi0_mat, Channel.NONE)

    @staticmethod
    def create_chi0_sum(
        g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
    ) -> "LocalSusceptibility":
        gchi0 = LocalSusceptibility.create_generalized_chi0(g_loc, frequency_shift)
        # gchi0 has a factor beta in front, that's why we divide by beta squared here
        return (1.0 / (config.beta * config.beta) * gchi0).contract_legs()

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

    @staticmethod
    def create_auxiliary_chi(
        gamma_r: "LocalIrreducibleVertex", gchi_0: "LocalSusceptibility", u_loc: LocalInteraction
    ) -> "LocalSusceptibility":
        chi_aux_mat = (~(~gchi_0 + gamma_r - u_loc.as_channel(gamma_r.channel) / (config.beta * config.beta))).mat

        return LocalSusceptibility(
            chi_aux_mat, gamma_r.channel, full_niw_range=gamma_r.full_niw_range, full_niv_range=gamma_r.full_niv_range
        )

    @staticmethod
    def create_physical_chi(
        chi_aux_contracted: "LocalSusceptibility", gchi0_sum: "LocalSusceptibility", u_loc: LocalInteraction
    ) -> "LocalSusceptibility":
        chi_phys_mat = (~(~(chi_aux_contracted + gchi0_sum) + u_loc.as_channel(chi_aux_contracted.channel))).mat
        return LocalSusceptibility(
            chi_phys_mat,
            chi_aux_contracted.channel,
            1,
            0,
            chi_aux_contracted.full_niw_range,
            chi_aux_contracted.full_niv_range,
        )

    @staticmethod
    def create_vrg(gchi_aux: "LocalSusceptibility", gchi0: "LocalSusceptibility") -> LocalThreePoint:
        gchi_aux_sum = gchi_aux.sum(axis=(-1,))
        vrg_mat = MFHelper.compress_last_two_frequency_dimensions_to_single_dimension(((~gchi0) @ gchi_aux_sum).mat)
        return LocalThreePoint(vrg_mat, gchi_aux.channel, 1, 1, gchi_aux.full_niw_range, gchi_aux.full_niv_range)
