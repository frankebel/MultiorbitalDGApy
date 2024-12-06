import gc
from copy import deepcopy

import numpy as np

import config
from matsubara_frequency_helper import MFHelper, FrequencyShift
from i_have_channel import Channel
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from interaction import LocalInteraction
from local_three_point import LocalThreePoint


def create_irreducible_vertex(gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    chi_tilde_mat = ((~gchi0) + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)).mat
    chi_tilde = LocalFourPoint(
        chi_tilde_mat,
        gchi_r.channel,
        full_niw_range=gchi0.full_niw_range,
        full_niv_range=gchi0.full_niv_range,
    )

    gammar_mat = ((~gchi_r) - chi_tilde + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)).mat
    return LocalFourPoint(
        gammar_mat,
        gchi_r.channel,
        full_niw_range=gchi_r.full_niw_range,
        full_niv_range=gchi_r.full_niv_range,
    )


def create_generalized_chi(g2: LocalFourPoint, g_loc: LocalGreensFunction) -> LocalFourPoint:
    """gchi_r:1/eV^3 = beta:1/eV * (G2_r:1/eV^2 - 2 * GG:1/eV^2 delta_dens delta_w0)"""
    chi = config.beta * g2

    if g2.channel == Channel.DENS:
        ggv_mat = _get_ggv_mat(g_loc, niv_slice=g2.niv)
        wn = MFHelper.get_wn_int(g2.niw)
        chi[:, :, :, :, wn == 0] = config.beta * (g2[:, :, :, :, wn == 0] - 2.0 * ggv_mat)

    return LocalFourPoint(
        chi.mat,
        chi.channel,
        full_niw_range=chi.full_niw_range,
        full_niv_range=chi.full_niv_range,
    )


def create_generalized_chi0(
    g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
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
    return LocalFourPoint(gchi0_mat, Channel.NONE)


def create_chi0_sum(
    g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    gchi0 = create_generalized_chi0(g_loc, frequency_shift)
    # gchi0 has a factor beta in front, that's why we divide by beta squared here
    return (1.0 / (config.beta * config.beta) * gchi0).contract_legs()


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


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    chi_aux_mat = (~(~gchi_0 + gamma_r - u_loc.as_channel(gamma_r.channel) / (config.beta * config.beta))).mat

    return LocalFourPoint(
        chi_aux_mat, gamma_r.channel, full_niw_range=gamma_r.full_niw_range, full_niv_range=gamma_r.full_niv_range
    )


def create_physical_chi(
    chi_aux_contracted: LocalFourPoint, gchi0_sum: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    chi_phys_mat = (~(~(chi_aux_contracted + gchi0_sum) + u_loc.as_channel(chi_aux_contracted.channel))).mat
    return LocalFourPoint(
        chi_phys_mat,
        chi_aux_contracted.channel,
        1,
        0,
        chi_aux_contracted.full_niw_range,
        chi_aux_contracted.full_niv_range,
    )


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalThreePoint:
    gchi_aux_sum = gchi_aux.sum_over_fermionic_dimensions(axis=(-1,))
    vrg_mat = MFHelper.compress_last_two_frequency_dimensions_to_single_dimension(((~gchi0) @ gchi_aux_sum).mat)
    return LocalThreePoint(vrg_mat, gchi_aux.channel, 1, 1, gchi_aux.full_niw_range, gchi_aux.full_niv_range)


def calculate_local_self_energy(
    g_loc: LocalGreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
) -> LocalSelfEnergy:
    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    gchi_magn = create_generalized_chi(g2_magn, g_loc)

    if config.do_plotting:
        gchi_dens.plot(omega=0, figure_name=f"Gchi_dens")
        gchi_magn.plot(omega=0, figure_name=f"Gchi_magn")

    gchi0 = create_generalized_chi0(g_loc)

    gamma_dens = create_irreducible_vertex(gchi_dens, gchi0, u_loc)
    gamma_magn = create_irreducible_vertex(gchi_magn, gchi0, u_loc)

    del gchi_dens, gchi_magn
    gc.collect()

    if config.do_plotting:
        gamma_dens_copy = deepcopy(gamma_dens)
        gamma_magn_copy = deepcopy(gamma_magn)

        gamma_dens_copy = gamma_dens_copy.cut_niv(min(config.niv, 2 * int(config.beta)))
        gamma_magn_copy = gamma_magn_copy.cut_niv(min(config.niv, 2 * int(config.beta)))

        gamma_dens_copy.plot(omega=0, figure_name="Gamma_dens")
        gamma_magn_copy.plot(omega=0, figure_name="Gamma_magn")

        gamma_dens_copy.plot(omega=10, figure_name="Gamma_dens")
        gamma_dens_copy.plot(omega=-10, figure_name="Gamma_dens")

        gamma_magn_copy.plot(omega=10, figure_name="Gamma_magn")
        gamma_magn_copy.plot(omega=-10, figure_name="Gamma_magn")

        del gamma_dens_copy, gamma_magn_copy
        gc.collect()

    gchi0_sum = create_chi0_sum(g_loc)

    gchi_aux_dens = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    gchi_aux_dens_contracted = gchi_aux_dens.contract_legs()

    gchi_aux_magn = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    gchi_aux_magn_contracted = gchi_aux_magn.contract_legs()

    chi_phys_dens = create_physical_chi(gchi_aux_dens_contracted, gchi0_sum, u_loc)
    chi_phys_magn = create_physical_chi(gchi_aux_magn_contracted, gchi0_sum, u_loc)

    vrg_dens = create_vrg(gchi_aux_dens, gchi0)
    vrg_magn = create_vrg(gchi_aux_magn, gchi0)

    return LocalSelfEnergy(np.array([]), g_loc.full_niv_range)
