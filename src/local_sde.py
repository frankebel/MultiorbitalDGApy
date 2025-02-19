import config

from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from greens_function import GreensFunction
from self_energy import SelfEnergy
from local_three_point import LocalThreePoint
from matsubara_frequencies import MFHelper, FrequencyShift
from memory_helper import MemoryHelper
from n_point_base import *


def create_gamma_r_v2(gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the irreducible vertex gamma_r with an rpa-approach to the urange.
    """
    u = 1.0 / config.sys.beta**2 * u_loc.as_channel(gchi_r.channel)
    chi_tilde_shell = ~(~gchi0 + u)
    chi_tilde_core = ~(chi_tilde_shell.cut_niv(config.box.niv))
    return config.sys.beta**2 * (~gchi_r - chi_tilde_core + u)


def create_gamma_r(gchi_r: LocalFourPoint, gchi0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the irreducible vertex gamma_r = (gchi_r)^(-1) - (gchi0)^(-1)
    """
    gamma_ll = config.sys.beta**2 * (~gchi_r - ~(gchi0.cut_niv(config.box.niv)))

    if config.box.niv_asympt == 0:
        return gamma_ll

    u_r = u_loc.as_channel(gchi_r.channel, config.sys.beta)
    gamma_lh, gamma_hl, gamma_hh = (
        np.tile(u_r.mat[..., None, None, None], (1,) * 4 + (2 * config.box.niw + 1,) + (2 * niv1,) + (2 * niv2,))
        for niv1, niv2 in [
            (config.box.niv, config.box.niv_full),
            (config.box.niv_full, config.box.niv),
            (config.box.niv_full, config.box.niv_full),
        ]
    )

    for gamma in (gamma_lh, gamma_hl, gamma_hh):
        gamma[
            ...,
            config.box.niv_asympt : config.box.niv_full + config.box.niv,
            config.box.niv_asympt : config.box.niv_full + config.box.niv,
        ] = 0

    gamma_lh = gamma_lh.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
        2 * config.box.niw + 1,
        config.sys.n_bands**2 * 2 * config.box.niv,
        config.sys.n_bands**2 * 2 * config.box.niv_full,
    )
    gamma_hl = gamma_hl.transpose(4, 0, 1, 5, 2, 3, 6).reshape(
        2 * config.box.niw + 1,
        config.sys.n_bands**2 * 2 * config.box.niv_full,
        config.sys.n_bands**2 * 2 * config.box.niv,
    )
    gamma_hh = LocalFourPoint(gamma_hh, gchi_r.channel)
    inner = (
        (~(gamma_hh + config.sys.beta**2 * (~gchi0)))
        .mat.transpose(4, 0, 1, 5, 2, 3, 6)
        .reshape(
            2 * config.box.niw + 1,
            config.sys.n_bands**2 * 2 * config.box.niv_full,
            config.sys.n_bands**2 * 2 * config.box.niv_full,
        )
    )
    correction = np.matmul(np.matmul(gamma_lh, inner), gamma_hl)
    MemoryHelper.delete(gamma_lh, gamma_hl, inner)

    compound_index_shape = (config.sys.n_bands, config.sys.n_bands, 2 * config.box.niv)
    correction = correction.reshape((2 * config.box.niw + 1,) + compound_index_shape * 2).transpose(1, 2, 4, 5, 0, 3, 6)

    gamma_hh[
        ...,
        config.box.niv_asympt : config.box.niv_full + config.box.niv,
        config.box.niv_asympt : config.box.niv_full + config.box.niv,
    ] = (
        gamma_ll.mat + correction
    )

    return gamma_hh


def create_gamma_0(u_loc: LocalInteraction, channel: Channel = Channel.DENS) -> LocalInteraction:
    """
    Returns the zero-th order vertex gamma_0 = (U_{abcd} - U_{adcb}) for a given channel.
    """
    u = u_loc.as_channel(channel) if u_loc.channel == channel.NONE else u_loc
    return u - u.permute_orbitals("abcd->adcb")


def create_generalized_chi(g2: LocalFourPoint, g_loc: GreensFunction) -> LocalFourPoint:
    """
    Returns the generalized susceptibility gchi_{r;lmm'l'}^{wvv'}:1/eV^3 = beta:1/eV * (G2_{r;lmm'l'}^{wvv'}:1/eV^2 - 2 * G_{ll'}^{v} G_{m'm}^{v}:1/eV^2 delta_dens delta_w0)
    """
    chi = config.sys.beta * g2

    if g2.channel == Channel.DENS:
        wn = MFHelper.wn(g2.niw)
        chi0_mat = _get_ggv_mat(g_loc, niv_slice=g2.niv)[:, :, :, :, np.newaxis, ...]
        chi[:, :, :, :, wn == 0, ...] -= 2.0 * chi0_mat

    return chi


def _get_ggv_mat(g_loc: GreensFunction, niv_slice: int = -1) -> np.ndarray:
    """
    Returns G_{ll'}^{v}:1/eV * G_{m'm}^{v}:1/eV
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
    return LocalFourPoint(-config.sys.beta * g_left_mat * g_right_mat, Channel.NONE, 1, 1)


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the auxiliary susceptibility gchi_aux_{r;lmm'l'} = ((gchi_{0;lmm'l'})^(-1) + gamma_{r;lmm'l'}-(u_{lmm'l'} - u_{ll'm'm})/beta^2)^(-1). See Eq. (3.68) in Paul Worm's thesis.
    """
    gamma_0 = create_gamma_0(u_loc, gamma_r.channel)
    return ~(~gchi_0 + (gamma_r - gamma_0) / config.sys.beta**2)


def create_physical_chi(gchi_r: LocalFourPoint) -> LocalFourPoint:
    """
    Returns the physical susceptibility chi_phys_{r;ll'}^{w} = 1/beta^2 [sum_v sum_{mm'} gchi_{r;lmm'l'}]. See Eq. (3.51) in Paul Worm's thesis.
    """
    return gchi_r.contract_legs(config.sys.beta)


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalThreePoint:
    """
    Returns the three-leg vertex vrg = beta * (gchi0)^(-1) * (sum_v gchi_aux). sum_v is performed over the fermionic
    frequency dimensions and includes a factor 1/beta. See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_sum = gchi_aux.sum_over_fermionic_dimensions(config.sys.beta, axis=(-1,))
    vrg_mat = (
        config.sys.beta * ((~gchi0) @ gchi_aux_sum).compress_last_two_frequency_dimensions_to_single_dimension().mat
    )
    return LocalThreePoint(vrg_mat, gchi_aux.channel, 1, 1, gchi_aux.full_niw_range, gchi_aux.full_niv_range)


def create_local_double_counting_kernel(gamma_r: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    eye = np.eye(2 * config.box.niv_full, dtype=np.complex128)[None, None, None, None, None, :, :]
    return -gamma_r @ ~((gamma_r @ gchi0) - eye)


def get_self_energy(
    vrg_dens: LocalThreePoint,
    gchi_dens: LocalFourPoint,
    g_loc: GreensFunction,
    u_loc: LocalInteraction,
) -> SelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation, see Paul Worm's thesis, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """
    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p
    g_1 = MFHelper.wn_slices_gen(g_loc.mat, config.box.niv, config.box.niw)
    deltas = np.einsum("io,lp->ilpo", np.eye(config.sys.n_bands), np.eye(config.sys.n_bands))

    gchi_dens = gchi_dens.sum_over_fermionic_dimensions(config.sys.beta, axis=(-1, -2))

    u_r = u_loc.as_channel(gchi_dens.channel)
    gamma_0 = create_gamma_0(u_r)
    inner_sum = np.einsum("abcd,ilbawv,dcpow->ilpowv", gamma_0.mat, vrg_dens.mat, gchi_dens.mat)
    inner = deltas[..., None, None] - vrg_dens.mat + inner_sum

    self_energy_mat = 1.0 / config.sys.beta * np.einsum("kjpo,ilpowv,lkwv->ijv", u_loc.mat, inner, g_1)
    hartree = np.einsum("abcd,dc->ab", u_loc.as_channel(Channel.DENS).mat, config.sys.occ)[..., None]

    return SelfEnergy(self_energy_mat + hartree)


def perform_local_schwinger_dyson(
    g_loc: GreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
) -> (SelfEnergy, LocalFourPoint, LocalFourPoint):
    """
    Performs the local Schwinger-Dyson equation calculation for the local self-energy.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
    """
    logger = config.logger

    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    MemoryHelper.delete(g2_dens)
    logger.log_info("Generalized susceptibilitiy (dens) done.")
    gchi_magn = create_generalized_chi(g2_magn, g_loc)
    MemoryHelper.delete(g2_magn)
    logger.log_info("Generalized susceptibilitiy (magn) done.")

    if config.output.do_plotting and config.current_rank == 0:
        gchi_dens.plot(omega=0, name=f"Gchi_dens", output_dir=config.output.output_path)
        gchi_magn.plot(omega=0, name=f"Gchi_magn", output_dir=config.output.output_path)
        logger.log_info("Generalized susceptibilities plotted.")

    gchi0 = create_generalized_chi0(g_loc)

    gamma_dens = create_gamma_r(gchi_dens, gchi0, u_loc)
    logger.log_info("Irreducible vertex (dens) done.")
    # gamma_dens = create_gamma_r_2(gchi_dens, gchi0, u_loc)
    gchi_aux_dens = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    logger.log_info("Auxiliary susceptibility (dens) done.")
    vrg_dens = create_vrg(gchi_aux_dens, gchi0)
    logger.log_info("Three-leg vertex (dens) done.")
    MemoryHelper.delete(gchi_aux_dens)
    f_dc_kernel_dens = create_local_double_counting_kernel(gamma_dens, gchi0)
    logger.log_info("Double-counting kernel (dens) done.")

    gamma_magn = create_gamma_r(gchi_magn, gchi0, u_loc)
    logger.log_info("Irreducible vertex (magn) done.")
    # gamma_magn = create_gamma_r_2(gchi_magn, gchi0, u_loc)
    gchi_aux_magn = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    logger.log_info("Auxiliary susceptibility (magn) done.")
    vrg_magn = create_vrg(gchi_aux_magn, gchi0)
    logger.log_info("Three-leg vertex (magn) done.")
    MemoryHelper.delete(gchi_aux_magn)
    f_dc_kernel_magn = create_local_double_counting_kernel(gamma_magn, gchi0)
    logger.log_info("Double-counting kernel (magn) done.")
    MemoryHelper.delete(gchi0)

    sigma = get_self_energy(vrg_dens, gchi_dens, g_loc, u_loc)

    logger.log_info("Self-energy done.")

    chi_dens_physical = create_physical_chi(gchi_dens)
    MemoryHelper.delete(gchi_dens)
    logger.log_info("Physical susceptibility (dens) done.")
    chi_magn_physical = create_physical_chi(gchi_magn)
    MemoryHelper.delete(gchi_magn)
    logger.log_info("Physical susceptibility (magn) done.")

    return (
        gamma_dens,
        gamma_magn,
        chi_dens_physical,
        chi_magn_physical,
        vrg_dens,
        vrg_magn,
        f_dc_kernel_dens,
        f_dc_kernel_magn,
        sigma,
    )
