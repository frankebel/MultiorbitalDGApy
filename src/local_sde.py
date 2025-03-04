import gc

import numpy as np

import config
from greens_function import GreensFunction
from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from matsubara_frequencies import MFHelper, FrequencyShift
from n_point_base import *
from self_energy import SelfEnergy


def create_generalized_chi(g2: LocalFourPoint, g_loc: GreensFunction) -> LocalFourPoint:
    """
    Returns the generalized susceptibility gchi_{r;lmm'l'}^{wvv'}:1/eV^3 = beta:1/eV * (G2_{r;lmm'l'}^{wvv'}:1/eV^2 - 2 * G_{ll'}^{v} G_{m'm}^{v}:1/eV^2 delta_dens delta_w0)
    """
    chi = config.sys.beta * g2

    if g2.channel == SpinChannel.DENS:
        wn = MFHelper.wn(g2.niw)
        ggv_mat = _get_ggv_mat(g_loc, niv_slice=g2.niv)[:, :, :, :, None, ...]
        chi[:, :, :, :, wn == 0, ...] -= 2.0 * config.sys.beta * ggv_mat

    return chi


def _get_ggv_mat(g_loc: GreensFunction, niv_slice: int = -1) -> np.ndarray:
    """
    Returns beta:1/eV * G_{ll'}^{v}:1/eV * G_{m'm}^{v}:1/eV
    """
    if niv_slice == -1:
        niv_slice = g_loc.niv
    g_loc_slice_mat = g_loc.mat[..., g_loc.niv - niv_slice : g_loc.niv + niv_slice]
    g_left_mat = g_loc_slice_mat[:, None, None, :, :, None] * np.eye(g_loc.n_bands)[None, :, :, None, None, None]
    g_right_mat = (
        np.swapaxes(g_loc_slice_mat, 0, 1)[None, :, :, None, None, :]
        * np.eye(g_loc.n_bands)[:, None, None, :, None, None]
    )
    return g_left_mat * g_right_mat


def create_generalized_chi0(
    g_loc: GreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    """
    Returns the generalized bare susceptibility gchi0_{lmm'l'}^{wvv}:1/eV^3 = -beta:1/eV * G_{ll'}^{v}:1/eV * G_{m'm}^{v-w}:1/eV
    """
    wn = MFHelper.wn(config.box.niw)
    iws, iws2 = np.array([MFHelper.get_frequency_shift(wn_i, frequency_shift) for wn_i in wn], dtype=int).T

    niv_range = np.arange(-config.box.niv_full, config.box.niv_full)
    g_left_mat = (
        g_loc.mat[:, None, None, :, g_loc.niv + niv_range[None, :] + iws[:, None]]
        * np.eye(g_loc.n_bands)[None, :, :, None, None, None]
    )
    g_right_mat = (
        g_loc.mat[None, :, :, None, g_loc.niv + niv_range[None, :] + iws2[:, None]]
        * np.eye(g_loc.n_bands)[:, None, None, :, None, None]
    )
    return LocalFourPoint(-config.sys.beta * g_left_mat * g_right_mat, SpinChannel.NONE, 1, 1)


def create_gamma_0(u_loc: LocalInteraction) -> LocalInteraction:
    r"""
    Returns the zero-th order vertex
    .. math:: \Gamma_{0;lmm'l'} = U_{lmm'l'} - U_{ll'm'm}.
    """
    return u_loc - u_loc.permute_orbitals("abcd->adcb")


def create_gamma_r(gchi_r: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the irreducible vertex
    .. math:: \Gamma_{r;lmm'l'}^{wvv'} = \beta^2 * [(\chi_{r;lmm'l'}^{wvv'})^{-1} - (\chi_{0;lmm'l'}^{wv})^{-1}]
    """
    return config.sys.beta**2 * (~gchi_r - ~(gchi0.cut_niv(config.box.niv)))


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    r"""
    Returns the auxiliary susceptibility
    .. math:: \chi^{*;wvv'}_{r;lmm'l'} = (\chi_{0;lmm'l'}^{-1} + (\Gamma_{r;lmm'l'}-\Gamma_{0;lmm'l'})/\beta^2)^{-1}.
    See Eq. (3.68) in Paul Worm's thesis.
    """
    return ~(~gchi_0 + (gamma_r - create_gamma_0(u_loc)) / config.sys.beta**2)


def create_physical_chi(gchi_r: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the physical susceptibility
    .. math:: \chi_{r;ll'}^{phys;w} = 1/\beta^2 [\sum_{vv'} \sum_{mm'} \chi_{r;lmm'l'}^{wvv'}].
    See Eq. (3.51) in Paul Worm's thesis.
    """
    return gchi_r.contract_legs(config.sys.beta)


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    r"""
    Returns the three-leg vertex
    .. math:: \gamma_{r;lmm'l'} = \beta * (\chi_{0;lmm'l'})^{-1} * (\sum_{v'} \chi^{*;wvv'}_{r;lmm'l'}).
    See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_sum = gchi_aux.sum_over_vn(config.sys.beta, axis=(-1,))
    return config.sys.beta * (~gchi0 @ gchi_aux_sum).compress_vn_dimensions()


def create_local_double_counting_kernel(gamma_r: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    eye = np.eye(2 * config.box.niv_full, dtype=np.complex128)[None, None, None, None, None, :, :]
    return -gamma_r @ ~((gamma_r @ gchi0) - eye)


def get_loc_self_energy_vrg(
    vrg_dens: LocalFourPoint, gchi_dens: LocalFourPoint, g_loc: GreensFunction, u_loc: LocalInteraction
) -> SelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation,
    see Paul Worm's thesis, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """
    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv, config.box.niw)
    deltas = np.einsum("io,lp->ilpo", np.eye(config.sys.n_bands), np.eye(config.sys.n_bands))
    gchi_dens = gchi_dens.sum_over_vn(config.sys.beta, axis=(-1, -2))
    inner = (vrg_dens @ create_gamma_0(u_loc) @ gchi_dens) - vrg_dens + deltas[..., None, None]
    self_energy_mat = 1.0 / config.sys.beta * u_loc.times("kjpo,ilpowv,lkwv->ijv", inner, g_1)
    hartree = u_loc.as_channel(SpinChannel.DENS).times("abcd,dc->ab", config.sys.occ)[..., None]
    return SelfEnergy(self_energy_mat + hartree)


def get_loc_self_energy_gamma_abinitio_dga(
    gamma_dens: LocalFourPoint, u_loc: LocalInteraction, g_loc: GreensFunction
) -> SelfEnergy:
    r"""
    Returns the local self-energy with the three-leg gamma from AbinitioDGA
    .. math:: \Sigma_{12}^{v} = -1/\beta \sum_w [ U_{a1bc} * \gamma_{cb2f}^{wv} * G_{af}^{w-v} ]
    """
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv, config.box.niw)
    hartree = np.einsum("abcd,dc->ab", u_loc.as_channel(SpinChannel.DENS).mat, config.sys.occ)[..., None]
    sigma = -1.0 / config.sys.beta * np.einsum("kjop,ilpowv,lkwv->ijv", u_loc.mat, gamma_dens.mat, g_1, optimize=True)
    return SelfEnergy(sigma + hartree)


def get_asympt_irr_gamma(u_loc: LocalInteraction, channel: SpinChannel = SpinChannel.DENS) -> LocalFourPoint:
    gamma_mat = 1.0 / config.sys.beta**2 * u_loc.as_channel(channel).mat
    gamma_mat = np.tile(gamma_mat[..., None], (1, 1, 1, 1, 2 * config.box.niw + 1))
    return LocalFourPoint(gamma_mat, channel, 1, 0)


def perform_local_schwinger_dyson(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    """
    ATTENTION: THIS IS PAUL'S ROUTINE
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
    """
    logger = config.logger

    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    del g2_dens
    gc.collect()
    logger.log_info("Local generalized susceptibility chi^wvv' (dens) done.")
    gchi_magn = create_generalized_chi(g2_magn, g_loc)
    del g2_magn
    gc.collect()
    logger.log_info("Local generalized susceptibility chi^wvv' (magn) done.")

    if config.output.do_plotting and config.current_rank == 0:
        gchi_dens.plot(omega=0, name=f"Gchi_dens", output_dir=config.output.output_path)
        gchi_magn.plot(omega=0, name=f"Gchi_magn", output_dir=config.output.output_path)
        logger.log_info("Local generalized susceptibilities plotted.")

    gchi0 = create_generalized_chi0(g_loc)

    gamma_dens = create_gamma_r(gchi_dens, gchi0)
    logger.log_info("Local irreducible vertex Gamma^wvv' (dens) done.")
    gchi_aux_dens = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    logger.log_info("Local auxiliary susceptibility chi^*wvv' (dens) done.")
    vrg_dens = create_vrg(gchi_aux_dens, gchi0)
    logger.log_info("Local three-leg vertex gamma^wv (dens) done.")
    del gchi_aux_dens
    gc.collect()

    gamma_magn = create_gamma_r(gchi_magn, gchi0)
    logger.log_info("Local irreducible vertex Gamma^wvv' (magn) done.")
    gchi_aux_magn = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    logger.log_info("Local auxiliary susceptibility chi^*wvv' (magn) done.")
    vrg_magn = create_vrg(gchi_aux_magn, gchi0)
    logger.log_info("Local three-leg vertex gamma^wv (magn) done.")
    del gchi_aux_magn, gchi0
    gc.collect()

    sigma = get_loc_self_energy_vrg(vrg_dens, gchi_dens, g_loc, u_loc)
    logger.log_info("Self-energy Sigma^v done.")

    chi_dens_physical = create_physical_chi(gchi_dens)
    logger.log_info("Local physical susceptibility chi^w (dens) done.")
    del gchi_dens
    gc.collect()

    chi_magn_physical = create_physical_chi(gchi_magn)
    logger.log_info("Local physical susceptibility chi^w (magn) done.")
    del gchi_magn
    gc.collect()

    return gamma_dens, gamma_magn, chi_dens_physical, chi_magn_physical, vrg_dens, vrg_magn, sigma


def perform_local_schwinger_dyson_abinitio_dga(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    """
    ATTENTION: THIS IS THE ABINITODGA ROUTINE!
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    """
    logger = config.logger

    gchi_dens_loc = create_generalized_chi(g2_dens, g_loc)
    del g2_dens
    gc.collect()
    logger.log_info("Generalized susceptibilitiy (dens) done.")
    gchi_magn_loc = create_generalized_chi(g2_magn, g_loc)
    del g2_magn
    gc.collect()
    logger.log_info("Generalized susceptibilitiy (magn) done.")

    gchi0_loc_full = create_generalized_chi0(g_loc)
    logger.log_info("Generalized bare susceptibilitiy done.")
    gchi0_core = gchi0_loc_full.cut_niv(config.box.niv)

    # 1 + chi0 * F_r = gchi_r * (chi0)^(-1) = 1 + gamma_r or
    # F_r = -beta^2 * [chi0^(-1) - chi0^(-1) chi_r chi0^(-1)]
    # gamma_r is NOT the irreducible vertex in channel r but rather the three-point vertex from AbinitioDGA
    gchi0_inv_core = ~gchi0_core

    f_dens_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_dens_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^w (dens) done.")
    f_magn_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_magn_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^w (magn) done.")
    del gchi0_inv_core
    gc.collect()

    one = LocalFourPoint.identity(config.sys.n_bands, config.box.niw, config.box.niv_full)
    # in most equations we need 1 + gamma_r so we add it here
    gamma_dens_loc = 1.0 / config.sys.beta * (gchi0_core @ f_dens_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_dens_loc = one + gamma_dens_loc
    logger.log_info("Local three-leg vertex gamma^w (dens) done.")

    gamma_magn_loc = 1.0 / config.sys.beta * (gchi0_core @ f_magn_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_magn_loc = one + gamma_magn_loc
    logger.log_info("Local three-leg vertex gamma^w (magn) done.")
    del gchi0_core, gamma_magn_loc
    gc.collect()

    sigma_loc = get_loc_self_energy_gamma_abinitio_dga(gamma_dens_loc, u_loc, g_loc)
    logger.log_info("Local self-energy done.")
    del gamma_dens_loc
    gc.collect()

    return (
        gchi_dens_loc,
        gchi_magn_loc,
        gchi0_loc_full,
        one_plus_gamma_dens_loc,
        one_plus_gamma_magn_loc,
        f_dens_loc,
        f_magn_loc,
        sigma_loc,
    )
