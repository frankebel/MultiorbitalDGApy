from copy import deepcopy

import numpy as np

import config
from i_have_channel import Channel
from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from local_three_point import LocalThreePoint
from matsubara_frequency_helper import MFHelper, FrequencyShift
from memory_helper import MemoryHelper


def create_irreducible_vertex(gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the irreducible vertex gamma_r = (gchi_r)^(-1) - chi_tilde + u_loc / beta^2
    """
    chi_tilde = (~gchi0) + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)
    return (~gchi_r) - chi_tilde + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)


def create_generalized_chi(g2: LocalFourPoint, g_loc: LocalGreensFunction) -> LocalFourPoint:
    """
    Returns the generalized susceptibility gchi_{r;lmm'l'}^{wvv'}:1/eV^3 = beta:1/eV * (G2_{r;lmm'l'}^{wvv'}:1/eV^2 - 2 * G_{ll'}^{v} G_{m'm}^{v}:1/eV^2 delta_dens delta_w0)
    """
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


def _get_ggv_mat(g_loc: LocalGreensFunction, niv_slice: int = -1) -> np.ndarray:
    """
    Returns G_{ll'}^{v}:1/eV * G_{m'm}^{v}:1/eV
    """
    if niv_slice == -1:
        niv_slice = g_loc.niv
    g_loc_slice_mat = g_loc.mat[..., g_loc.niv - niv_slice : g_loc.niv + niv_slice]
    eye_bands = np.eye(g_loc.n_bands)
    g_left_mat = g_loc_slice_mat[:, None, None, :, :, None] * eye_bands[None, :, :, None, None, None]
    g_right_mat = (
        np.swapaxes(g_loc_slice_mat, 0, 1)[None, :, :, None, None, :] * eye_bands[None, None, None, :, :, None]
    )
    return g_left_mat * g_right_mat


def create_generalized_chi0(
    g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    """
    Returns the generalized bare susceptibility gchi0_{lmm'l'}^{wvv}:1/eV^3 = -beta:1/eV * G_{ll'}^{v}:1/eV * G_{m'm}^{v-w}:1/eV
    """
    gchi0_mat = np.empty((g_loc.n_bands,) * 4 + (2 * config.niw + 1, 2 * config.niv), dtype=np.complex64)
    eye_bands = np.eye(g_loc.n_bands)

    wn = MFHelper.get_wn_int(config.niw)
    for index, current_wn in enumerate(wn):
        iws, iws2 = MFHelper.get_frequency_shift(current_wn, frequency_shift)

        # this is basically the same as _get_ggv_mat, but I don't know how to avoid the code duplication in a smart way
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

    return LocalFourPoint(gchi0_mat, Channel.NONE, 1, 1).extend_last_frequency_axis_to_diagonal()


def create_chi0_sum(
    g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    """
    Returns the sum over the generalized bare susceptibility gchi0_{lmm'l'}^{w}:eV = -1/beta:eV * (1/beta^2:eV^2 sum_v G_{ll'}^{v}:1/eV * G_{m'm}^{v-w}:1/eV)
    """
    gchi0 = create_generalized_chi0(g_loc, frequency_shift)
    # gchi0 has a factor beta in front, that's why we divide by beta squared here
    return (1.0 / (config.beta * config.beta) * gchi0).contract_legs(config.beta)


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the auxiliary susceptibility gchi_aux = (gamma_r + (gchi_0)^(-1) - u_loc / beta^2)^(-1)
    """
    return ~(gamma_r + ~gchi_0 - u_loc.as_channel(gamma_r.channel) / (config.beta * config.beta))


def create_physical_chi(
    chi_aux_contracted: LocalFourPoint, gchi0_sum: LocalFourPoint, u_loc: LocalInteraction
) -> LocalFourPoint:
    """
    Returns the physical susceptibility chi_phys = (([sum_v gchi_aux] - [sum_v gchi0])^(-1) + u_loc)^(-1). sum_v is performed over the fermionic frequency dimensions and includes a factor 1/beta^(n_dim).
    """
    return ~(~(chi_aux_contracted - gchi0_sum) + u_loc.as_channel(chi_aux_contracted.channel))


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalThreePoint:
    """
    Returns the three-leg vertex vrg = beta * (gchi0)^(-1) * (sum_v gchi_aux). sum_v is performed over the fermionic frequency dimensions and includes a factor 1/beta^2.
    """
    gchi_aux_sum = gchi_aux.sum_over_fermionic_dimensions(config.beta, axis=(-1,))
    vrg_mat = config.beta * ((~gchi0) @ gchi_aux_sum).compress_last_two_frequency_dimensions_to_single_dimension().mat
    return LocalThreePoint(vrg_mat, gchi_aux.channel, 1, 1, gchi_aux.full_niw_range, gchi_aux.full_niv_range)


def get_self_energy(
    vrg_dens: LocalThreePoint,
    chi_phys_dens: LocalFourPoint,
    g_loc: LocalGreensFunction,
    u_loc: LocalInteraction,
) -> LocalSelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation, see Paul Worm's thesis, P. 49, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """

    n_bands = vrg_dens.n_bands
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, vrg_dens.niv, vrg_dens.niw)
    deltas = np.einsum("io,lp->ilpo", np.eye(n_bands), np.eye(n_bands))

    inner_sum_left = np.einsum("abcd,ilbawv,dcpow->ilpowv", u_loc.mat, vrg_dens.mat, chi_phys_dens.mat)
    inner_sum_right = np.einsum("adcb,ilbawv,dcpow->ilpowv", u_loc.mat, vrg_dens.mat, chi_phys_dens.mat)
    inner = deltas - vrg_dens.mat + inner_sum_left - inner_sum_right

    self_energy_mat = 1.0 / config.beta * np.einsum("kjpo,ilpowv,lkwv->ijv", u_loc.mat, inner, g_1)
    self_energy_mat += get_hartree_fock(u_loc, config.n_dmft)
    return LocalSelfEnergy(self_energy_mat)


def get_hartree_fock(u_loc: LocalInteraction, n: float) -> np.ndarray:
    """
    Computes the local Hartree-Fock contribution with the n obtained from DMFT.
    """
    u = u_loc.as_channel(Channel.DENS).mat
    n = n * np.eye(u_loc.n_bands)
    hartree = 0.5 * np.einsum("abcd,dc->ab", u, n)
    fock = np.zeros_like(hartree)
    return hartree + fock


def perform_schwinger_dyson(
    g_loc: LocalGreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
) -> (LocalSelfEnergy, LocalFourPoint, LocalFourPoint):
    """
    Performs the Schwinger-Dyson equation calculation for the local self-energy.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
    """
    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    gchi_magn = create_generalized_chi(g2_magn, g_loc)

    if config.do_plotting:
        gchi_dens.plot(omega=0, name=f"Gchi_dens")
        gchi_magn.plot(omega=0, name=f"Gchi_magn")

    gchi0 = create_generalized_chi0(g_loc)

    gamma_dens = create_irreducible_vertex(gchi_dens, gchi0, u_loc)
    gamma_magn = create_irreducible_vertex(gchi_magn, gchi0, u_loc)

    MemoryHelper.delete(gchi_dens, gchi_magn)

    if config.do_plotting:
        gamma_dens_copy = deepcopy(gamma_dens)
        gamma_magn_copy = deepcopy(gamma_magn)

        gamma_dens_copy = gamma_dens_copy.cut_niv(min(config.niv, 2 * int(config.beta)))
        gamma_magn_copy = gamma_magn_copy.cut_niv(min(config.niv, 2 * int(config.beta)))

        gamma_dens_copy.plot(omega=0, name="Gamma_dens")
        gamma_magn_copy.plot(omega=0, name="Gamma_magn")

        gamma_dens_copy.plot(omega=10, name="Gamma_dens")
        gamma_dens_copy.plot(omega=-10, name="Gamma_dens")

        gamma_magn_copy.plot(omega=10, name="Gamma_magn")
        gamma_magn_copy.plot(omega=-10, name="Gamma_magn")

        MemoryHelper.delete(gamma_dens_copy, gamma_magn_copy)

    gchi0_sum = create_chi0_sum(g_loc)

    gchi_aux_dens = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    gchi_aux_dens_contracted = gchi_aux_dens.contract_legs(config.beta)

    gchi_aux_magn = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    gchi_aux_magn_contracted = gchi_aux_magn.contract_legs(config.beta)

    chi_dens = create_physical_chi(gchi_aux_dens_contracted, gchi0_sum, u_loc)
    chi_magn = create_physical_chi(gchi_aux_magn_contracted, gchi0_sum, u_loc)

    MemoryHelper.delete(gchi0_sum, gchi_aux_magn_contracted)

    vrg_dens = create_vrg(gchi_aux_dens, gchi0)
    vrg_magn = create_vrg(gchi_aux_magn, gchi0)

    MemoryHelper.delete(gchi0, gchi_aux_dens, gchi_aux_magn)

    sigma = get_self_energy(vrg_dens, gchi_aux_dens_contracted, g_loc, u_loc)

    return gamma_dens, gamma_magn, chi_dens, chi_magn, vrg_dens, vrg_magn, sigma
