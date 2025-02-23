import gc

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

    if g2.channel == Channel.DENS:
        wn = MFHelper.wn(g2.niw)
        chi0_mat = _get_ggv_mat(g_loc, niv_slice=g2.niv)[:, :, :, :, None, ...]
        chi[:, :, :, :, wn == 0, ...] -= 2.0 * chi0_mat

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
    return config.sys.beta * g_left_mat * g_right_mat


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
    return LocalFourPoint(-config.sys.beta * g_left_mat * g_right_mat, Channel.NONE, 1, 1)


def get_loc_self_energy_gamma(gamma_dens: LocalFourPoint, u_loc: LocalInteraction, g_loc: GreensFunction) -> SelfEnergy:
    r"""
    Returns the local self-energy
    .. math:: \Sigma_{12}^{v} = -1/\beta \sum_w [ U_{a1bc} * \gamma_{cb2f}^{wv} * G_{af}^{w-v} ]
    """
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv, config.box.niw)
    hartree = np.einsum("abcd,dc->ab", u_loc.as_channel(Channel.DENS).mat, config.sys.occ)[..., None]
    sigma = -1.0 / config.sys.beta * np.einsum("kjop,ilpowv,lkwv->ijv", u_loc.mat, gamma_dens.mat, g_1, optimize=True)
    return SelfEnergy(sigma + hartree)


def get_asympt_irr_gamma(u_loc: LocalInteraction, channel: Channel = Channel.DENS) -> LocalFourPoint:
    gamma_mat = 1.0 / config.sys.beta**2 * u_loc.as_channel(channel).mat
    gamma_mat = np.tile(gamma_mat[..., None], (1, 1, 1, 1, 2 * config.box.niw + 1))
    return LocalFourPoint(gamma_mat, channel, 1, 0)


def perform_local_schwinger_dyson(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
):
    """
    Performs the local Schwinger-Dyson equation calculation for the local (DMFT) self-energy for sanity checks.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
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
    gchi0_inv_core = ~gchi0_core
    one_mat_loc = np.einsum("ac,bd->abcd", np.eye(config.sys.n_bands), np.eye(config.sys.n_bands))[..., None, None]

    f_dens_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_dens_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^w (dens) done.")
    f_magn_loc = -config.sys.beta**2 * (gchi0_inv_core - gchi0_inv_core @ gchi_magn_loc @ gchi0_inv_core)
    logger.log_info("Local full vertex F^w (magn) done.")
    del gchi0_inv_core
    gc.collect()

    """
    irr_gamma_urange_dens = get_asympt_irr_gamma(u_loc, Channel.DENS)
    irr_gamma_urange_magn = get_asympt_irr_gamma(u_loc, Channel.MAGN)

    test = (irr_gamma_urange_dens @ gchi0_loc_full).extend_vn_dimension() + one_mat_loc[..., None]
    test = ~test

    f_dens_loc_asympt_mat = -(
        (
            ~(
                (irr_gamma_urange_dens @ gchi0_loc_full).extend_vn_dimension()
                + config.sys.beta**2 * one_mat_loc[..., None]
            )
        )
        @ irr_gamma_urange_dens
    )
    
    f_magn_loc_asympt_mat = (
        config.sys.beta**2
        * (~(1.0 / config.sys.beta * (irr_gamma_urange_magn @ gchi0_loc_full) + one_mat_loc))
        @ irr_gamma_urange_magn
    )
    f_dens_loc = f_dens_loc.padding_along_vn(f_dens_loc_asympt_mat)
    """

    # in most equations we need 1 + gamma_r so we add it here
    gamma_dens_loc = 1.0 / config.sys.beta * (gchi0_core @ f_dens_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_dens_loc = gamma_dens_loc + one_mat_loc
    logger.log_info("Local three-leg vertex gamma^w (dens) done.")

    gamma_magn_loc = 1.0 / config.sys.beta * (gchi0_core @ f_magn_loc).sum_over_vn(config.sys.beta, axis=(-2,))
    one_plus_gamma_magn_loc = gamma_magn_loc + one_mat_loc
    logger.log_info("Local three-leg vertex gamma^w (magn) done.")
    del gchi0_core, gamma_magn_loc
    gc.collect()

    sigma_loc = get_loc_self_energy_gamma(gamma_dens_loc, u_loc, g_loc)
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
