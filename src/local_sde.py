import config

from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from local_three_point import LocalThreePoint
from matsubara_frequencies import MFHelper, FrequencyShift
from memory_helper import MemoryHelper
from n_point_base import *


def create_irreducible_vertex(gchi_r: LocalFourPoint, gchi0: LocalFourPoint) -> LocalFourPoint:
    """
    Returns the irreducible vertex gamma_r = (gchi_r)^(-1) - (gchi0)^(-1)
    """
    return ~gchi_r - ~gchi0


def create_generalized_chi(g2: LocalFourPoint, g_loc: LocalGreensFunction) -> LocalFourPoint:
    """
    Returns the generalized susceptibility gchi_{r;lmm'l'}^{wvv'}:1/eV^3 = beta:1/eV * (G2_{r;lmm'l'}^{wvv'}:1/eV^2 - 2 * G_{ll'}^{v} G_{m'm}^{v}:1/eV^2 delta_dens delta_w0)
    """
    chi = config.sys.beta * g2
    chi0_mat = _get_ggv_mat(g_loc, niv_slice=g2.niv)

    """
    if g2.channel == Channel.DENS:
        wn = MFHelper.wn(g2.niw)
        chi[:, :, :, :, wn == 0, ...] = config.beta * (
            g2[:, :, :, :, wn == 0, ...] - 2.0 * chi0_mat[:, :, :, :, wn == 0, ...]
        )
    """

    # just for testing!
    if g2.channel == Channel.DENS:
        wn = MFHelper.wn(g2.niw)
        chi[:, :, :, :, wn == 0, ...] = config.sys.beta * (
            g2[:, :, :, :, wn == 0, ...] - 2.0 * chi0_mat[0, 0, 0, 0, wn == 0, ...]
        )

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
    g_left_mat = g_loc_slice_mat[:, None, None, :, :, None] * np.eye(g_loc.n_bands)[None, :, :, None, None, None]
    g_right_mat = (
        np.swapaxes(g_loc_slice_mat, 0, 1)[None, :, :, None, None, :]
        * np.eye(g_loc.n_bands)[:, None, None, :, None, None]
    )
    ggv_mat = g_left_mat * g_right_mat
    ggv_mat = ggv_mat[:, :, :, :, np.newaxis, ...]
    return np.tile(ggv_mat, (1, 1, 1, 1, 2 * config.box.niw + 1, 1, 1))


def create_generalized_chi0(
    g_loc: LocalGreensFunction, frequency_shift: FrequencyShift = FrequencyShift.MINUS
) -> LocalFourPoint:
    """
    Returns the generalized bare susceptibility gchi0_{lmm'l'}^{wvv}:1/eV^3 = -beta:1/eV * G_{ll'}^{v}:1/eV * G_{m'm}^{v-w}:1/eV
    """
    gchi0_mat = np.empty((g_loc.n_bands,) * 4 + (2 * config.box.niw + 1, 2 * config.box.niv), dtype=np.complex64)

    wn = MFHelper.wn(config.box.niw)
    for index, current_wn in enumerate(wn):
        iws, iws2 = MFHelper.get_frequency_shift(current_wn, frequency_shift)

        # this is basically the same as _get_ggv_mat, but I don't know how to avoid the code duplication in a smart way
        g_left_mat = (
            g_loc.mat[..., g_loc.niv - config.box.niv + iws : g_loc.niv + config.box.niv + iws][:, None, None, :, :]
            * np.eye(g_loc.n_bands)[None, :, :, None, None]
        )
        g_right_mat = (
            np.swapaxes(g_loc.mat, 0, 1)[..., g_loc.niv - config.box.niv + iws2 : g_loc.niv + config.box.niv + iws2][
                None, :, :, None, :
            ]
            * np.eye(g_loc.n_bands)[:, None, None, :, None]
        )

        gchi0_mat[..., index, :] = -config.sys.beta * g_left_mat * g_right_mat

    return LocalFourPoint(gchi0_mat, Channel.NONE, 1, 1).extend_last_frequency_axis_to_diagonal()


def create_auxiliary_chi(gamma_r: LocalFourPoint, gchi_0: LocalFourPoint, u_loc: LocalInteraction) -> LocalFourPoint:
    """
    Returns the auxiliary susceptibility gchi_aux_{r;lmm'l'} = ((gchi_{0;lmm'l'})^(-1) + gamma_{r;lmm'l'}-(u_{lmm'l'} - u_{ll'm'm})/beta^2)^(-1). See Eq. (3.68) in Paul Worm's thesis.
    """
    u = u_loc.as_channel(gamma_r.channel)
    return ~(~gchi_0 + gamma_r - (u - u.permute_orbitals("abcd->adcb")) / config.sys.beta**2)


def create_physical_chi(gchi_r: LocalFourPoint) -> LocalFourPoint:
    """
    Returns the physical susceptibility chi_phys_{r;ll'}^{w} = 1/beta^2 [sum_v sum_{mm'} gchi_{r;lmm'l'}]. See Eq. (3.51) in Paul Worm's thesis.
    """
    return gchi_r.contract_legs(config.sys.beta).sum_over_orbitals("abcd->ad")


def create_vrg(gchi_aux: LocalFourPoint, gchi0: LocalFourPoint) -> LocalThreePoint:
    """
    Returns the three-leg vertex vrg = beta * (gchi0)^(-1) * (sum_v gchi_aux). sum_v is performed over the fermionic
    frequency dimensions and includes a factor 1/beta^2. See Eq. (3.71) in Paul Worm's thesis.
    """
    gchi_aux_sum = gchi_aux.sum_over_fermionic_dimensions(config.sys.beta, axis=(-1,))
    vrg_mat = (
        config.sys.beta * ((~gchi0) @ gchi_aux_sum).compress_last_two_frequency_dimensions_to_single_dimension().mat
    )
    return LocalThreePoint(vrg_mat, gchi_aux.channel, 1, 1, gchi_aux.full_niw_range, gchi_aux.full_niv_range)


def get_self_energy(
    vrg_dens: LocalThreePoint,
    gchi_dens: LocalFourPoint,
    g_loc: LocalGreensFunction,
    u_loc: LocalInteraction,
) -> LocalSelfEnergy:
    """
    Performs the local self-energy calculation using the Schwinger-Dyson equation, see Paul Worm's thesis, Eq. (3.70) and Anna Galler's Thesis, P. 76 ff.
    """
    n_bands = vrg_dens.n_bands

    # 1=i, 2=j, 3=k, 4=l, 7=o, 8=p

    g_1 = MFHelper.wn_slices_gen(g_loc.mat, vrg_dens.niv, vrg_dens.niw)
    deltas = np.einsum("io,lp->ilpo", np.eye(n_bands), np.eye(n_bands))

    gchi_dens = gchi_dens.contract_legs(config.sys.beta)

    inner_sum_left = np.einsum("abcd,ilbawv,dcpow->ilpowv", u_loc.mat, vrg_dens.mat, gchi_dens.mat)
    inner_sum_right = np.einsum("adcb,ilbawv,dcpow->ilpowv", u_loc.mat, vrg_dens.mat, gchi_dens.mat)
    inner = deltas[..., np.newaxis, np.newaxis] - vrg_dens.mat + inner_sum_left - inner_sum_right

    self_energy_mat = 1.0 / config.sys.beta * np.einsum("kjpo,ilpowv,lkwv->ijv", u_loc.mat, inner, g_1)

    hartree = np.einsum("abcd,bd->ac", u_loc.mat, config.sys.occ)
    self_energy_mat += hartree[..., np.newaxis]

    return LocalSelfEnergy(self_energy_mat)


def perform_local_schwinger_dyson(
    g_loc: LocalGreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
) -> (LocalSelfEnergy, LocalFourPoint, LocalFourPoint):
    """
    Performs the local Schwinger-Dyson equation calculation for the local self-energy.
    Includes the calculation of the three-leg vertices, (auxiliary/bare/physical) susceptibilities and the irreducible vertices.
    """
    gchi_dens = create_generalized_chi(g2_dens, g_loc)
    gchi_magn = create_generalized_chi(g2_magn, g_loc)

    if config.output.do_plotting:
        gchi_dens.plot(omega=0, name=f"Gchi_dens")
        gchi_magn.plot(omega=0, name=f"Gchi_magn")

    gchi0 = create_generalized_chi0(g_loc)

    # testing block
    # this will be removed later
    gchi_dens_copy = deepcopy(gchi0)
    gchi_dens_copy.mat[0, 0, 0, 0, ...] = gchi_dens.mat[0, 0, 0, 0, ...]
    gchi_dens = deepcopy(gchi_dens_copy)
    gchi_dens._channel = Channel.DENS
    MemoryHelper.delete(gchi_dens_copy)

    gchi_magn_copy = deepcopy(gchi0)
    gchi_magn_copy.mat[0, 0, 0, 0, ...] = gchi_magn.mat[0, 0, 0, 0, ...]
    gchi_magn = deepcopy(gchi_magn_copy)
    gchi_magn._channel = Channel.MAGN
    MemoryHelper.delete(gchi_magn_copy)

    assert np.allclose(gchi_dens[0, 0, 0, 0], gchi0[0, 0, 0, 0]) is False, "Nooo"
    assert np.allclose(gchi_magn[0, 0, 0, 0], gchi0[0, 0, 0, 0]) is False, "Nooo"
    assert np.allclose(gchi_dens[1, 1, 1, 1], gchi0[1, 1, 1, 1]), "Nooo"
    # endtesting block

    gamma_dens = create_irreducible_vertex(gchi_dens, gchi0)
    gamma_magn = create_irreducible_vertex(gchi_magn, gchi0)

    # testing block
    test = gamma_dens[0, 0, 0, 0, ...]
    test1 = gamma_dens[1, 1, 1, 1, ...]
    test1_zero = np.zeros_like(test1)
    res = np.allclose(test, test1_zero)
    assert res is False, "Shit"
    res = np.allclose(test1, test1_zero)
    assert res is True, "Shit"
    # endtesting block

    gchi_aux_dens = create_auxiliary_chi(gamma_dens, gchi0, u_loc)
    vrg_dens = create_vrg(gchi_aux_dens, gchi0)
    MemoryHelper.delete(gchi_aux_dens)

    gchi_aux_magn = create_auxiliary_chi(gamma_magn, gchi0, u_loc)
    vrg_magn = create_vrg(gchi_aux_magn, gchi0)
    MemoryHelper.delete(gchi0, gchi_aux_magn)

    sigma = get_self_energy(vrg_dens, gchi_dens, g_loc, u_loc)

    chi_dens_physical = create_physical_chi(gchi_dens)
    MemoryHelper.delete(gchi_dens)
    chi_magn_physical = create_physical_chi(gchi_magn)
    MemoryHelper.delete(gchi_magn)

    return gamma_dens, gamma_magn, chi_dens_physical, chi_magn_physical, vrg_dens, vrg_magn, sigma
