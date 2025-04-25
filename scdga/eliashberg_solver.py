import os
from copy import deepcopy

import mpi4py.MPI as MPI
import numpy as np
import scipy as sp

import scdga.config as config
from scdga.bubble_gen import BubbleGenerator
from scdga.four_point import FourPoint
from scdga.gap_function import GapFunction
from scdga.greens_function import GreensFunction
from scdga.interaction import LocalInteraction, Interaction
from scdga.local_four_point import LocalFourPoint
from scdga.matsubara_frequencies import MFHelper
from scdga.mpi_distributor import MpiDistributor
from scdga.n_point_base import SpinChannel, FrequencyNotation


def delete_files(filepath: str, *args) -> None:
    """
    Delete files in the given directory. If the file is not found, it will be ignored.
    """
    for name in args:
        if not isinstance(name, str):
            raise TypeError(f"Expected string, got {type(name)}.")
        full_path = os.path.join(filepath, name)
        if os.path.isfile(full_path):
            try:
                os.remove(full_path)
            except OSError:
                config.logger.log_info(f"Error deleting file: {name}.")


def gather_save_scatter(f_q_r: FourPoint, file_path: str, mpi_dist_irrk: MpiDistributor) -> FourPoint:
    """
    Gather the vertex function from all ranks, save it to a file, and scatter it back to the original ranks.
    """
    f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)

    if mpi_dist_irrk.my_rank == 0:
        f_q_r.save(output_dir=file_path, name=f"f_{f_q_r.channel.value}_irrq")

    f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
    return f_q_r


def create_full_vertex_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, comm: MPI.Comm
) -> FourPoint:
    """
    Calculate the full vertex function in the given channel. See Eq. (3.139) in my thesis.
    """
    logger = config.logger
    logger.log_info(f"Starting to calculate the full {channel.value} vertex.")

    gchi0_q_inv = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy"), num_vn_dimensions=1
    )
    gchi_aux_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy"), channel=channel
    )
    logger.log_info(f"Loaded gchi0_q_inv and gchi_aux_q_{gchi_aux_q_r.channel.value} from files.")

    f_q_r = config.sys.beta**2 * (gchi0_q_inv - gchi0_q_inv @ gchi_aux_q_r @ gchi0_q_inv)
    del gchi0_q_inv, gchi_aux_q_r
    logger.log_info(f"Calculated first part of full {f_q_r.channel.value} vertex.")

    delete_files(config.output.eliashberg_path, f"gchi_aux_q_{channel.value}_rank_{comm.rank}.npy")

    vrg_q_r = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"vrg_q_{channel.value}_rank_{comm.rank}.npy"),
        channel=channel,
        num_vn_dimensions=1,
    )
    gchi_aux_q_r_sum = FourPoint.load(
        os.path.join(config.output.eliashberg_path, f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy"),
        channel=channel,
        num_vn_dimensions=0,
    )
    logger.log_info(
        f"Loaded vrg_q_{vrg_q_r.channel.value} and gchi_aux_q_{gchi_aux_q_r_sum.channel.value}_sum from files."
    )

    u = u_loc.as_channel(channel) + v_nonloc.as_channel(channel)
    u_vrg_mul = u @ (vrg_q_r * vrg_q_r)
    del vrg_q_r
    f_q_r += u_vrg_mul - u @ gchi_aux_q_r_sum @ u_vrg_mul
    del u, u_vrg_mul, gchi_aux_q_r_sum

    logger.log_info(f"Calculated second part of full {f_q_r.channel.value} vertex.")

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{channel.value}_rank_{comm.rank}.npy",
        f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy",
    )

    return f_q_r


def transform_vertex_ph_to_pp_w0(f_q_r: LocalFourPoint, niv_pp: int) -> LocalFourPoint | FourPoint:
    """
    Transform the vertex function from particle-hole notation to particle-particle notation. This is done by
    flipping the last Matsubara frequency to get v, -v' and then applying the necessary condition of w = v-v'.
    """
    is_local = not isinstance(f_q_r, FourPoint)

    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]
    f_q_r_flip = f_q_r.cut_niv(niv_pp).to_full_niw_range().flip_frequency_axis(-1)
    del f_q_r
    f_q_r_pp_mat = np.zeros((*f_q_r_flip.current_shape[:-3], 2 * niv_pp, 2 * niv_pp), dtype=f_q_r_flip.mat.dtype)
    for idx, w in enumerate(MFHelper.wn(config.box.niw_core)):
        f_q_r_pp_mat[..., omega == w] = -f_q_r_flip[..., idx, omega == w]

    if is_local:
        return LocalFourPoint(f_q_r_pp_mat, f_q_r_flip.channel, 0, frequency_notation=FrequencyNotation.PP)
    return FourPoint(
        f_q_r_pp_mat, f_q_r_flip.channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP
    )


def calculate_full_vertex_pp_w0(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, niv_pp: int, mpi_dist_irrk: MpiDistributor
):
    """
    Calculates the full vertex function in PH notation and transforms it to PP notation.
    For the calculation of F, see Eq. (3.140a) and Eq. (3.140b) in my thesis.
    """
    logger = config.logger

    group_size = max(mpi_dist_irrk.comm.size // 3, 1)
    color = mpi_dist_irrk.comm.rank // group_size
    sub_comm = mpi_dist_irrk.comm.Split(color, mpi_dist_irrk.comm.rank)

    # we are splitting the comm into three groups to calculate the full vertex only for a small subset of the
    # irreducible BZ at a time to save memory.
    f_q_r = None
    for i in range(sub_comm.size):
        if sub_comm.rank == i:
            f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, channel, mpi_dist_irrk.comm)
        sub_comm.Barrier()
    sub_comm.Free()
    logger.log_info(f"Full momentum-dependent ladder-vertex ({channel.value}) calculated.")
    logger.log_memory_usage(f"Full vertex ({channel.value})", f_q_r, mpi_dist_irrk.comm.size)

    if config.output.save_fq:
        f_q_r = gather_save_scatter(f_q_r, config.output.output_path, mpi_dist_irrk)
        config.logger.log_info(f"Saved full {f_q_r.channel.value} vertex in the irreducible BZ to file.")

    return transform_vertex_ph_to_pp_w0(f_q_r, niv_pp)


def get_initial_gap_function(shape: tuple, channel: SpinChannel) -> np.ndarray:
    """
    Generates the initial gap function based on the specified shape, spin channel,
    and symmetry settings from the configuration. Depending on the symmetry and
    spin channel, it initializes the gap function with appropriate properties.
    """
    if channel not in {SpinChannel.SING, SpinChannel.TRIP}:
        raise ValueError("Channel must be either SING or TRIP.")

    gap0 = np.zeros(shape, dtype=np.complex64)
    niv = shape[-1] // 2
    k_grid = config.lattice.k_grid.grid

    symm = {
        "d-wave": lambda k: -np.cos(k[0])[:, None, None] + np.cos(k[1])[None, :, None],
        "p-wave-x": lambda k: np.sin(k[0])[:, None, None],
        "p-wave-y": lambda k: np.sin(k[1])[None, :, None],
    }

    if config.eliashberg.symmetry in symm:
        gap0[..., niv:] = np.repeat(symm[config.eliashberg.symmetry](k_grid)[:, :, :, None, None, None], niv, axis=-1)
    else:
        gap0 = np.random.random_sample(shape)

    v_sym = {
        "d-wave": "even" if channel == SpinChannel.SING else "odd",
        "p-wave-x": "odd" if channel == SpinChannel.SING else "even",
        "p-wave-y": "odd" if channel == SpinChannel.SING else "even",
    }.get(config.eliashberg.symmetry, "")

    if v_sym in {"even", "odd"}:
        gap0[..., :niv] = gap0[..., niv:] if v_sym == "even" else -gap0[..., niv:]
    else:
        gap0 = np.random.random_sample(shape)

    return gap0


def solve_eliashberg_lanczos(gamma_q_r_pp: FourPoint, gchi0_q0_pp: FourPoint):
    """
    Solves the Eliashberg equation for the superconducting eigenvalue and gap function using an
    Implicitly Restarted Lanczos Method. Returns the first n_eig eigenvalues and eigenvectors and the maximum
    eigenvalue and corresponding eigenvector in two separate lists, one for the lambdas, one for the gaps.
    """
    logger = config.logger

    gamma_q_r_pp = gamma_q_r_pp.map_to_full_bz(
        config.lattice.q_grid.irrk_inv, config.lattice.q_grid.nk
    ).decompress_q_dimension()
    logger.log_info(f"Mapped Gamma_pp ({gamma_q_r_pp.channel.value}) to full BZ.")
    logger.log_memory_usage(f"Gamma_pp_{gamma_q_r_pp.channel.value}", gamma_q_r_pp, 1)

    sign = 1 if gamma_q_r_pp.channel == SpinChannel.SING else -1

    gamma_x = sign * gamma_q_r_pp.fft()
    logger.log_info("Fourier-transformed Gamma_pp.")
    gamma_x_flipped = gamma_x.flip_momentum_axis().flip_frequency_axis(-2)

    gap_shape = gamma_q_r_pp.nq + 2 * (gamma_q_r_pp.n_bands,) + (2 * gamma_q_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    gap0 = get_initial_gap_function(gap_shape, gamma_q_r_pp.channel)
    logger.log_info(
        f"Initialized the gap function as {config.eliashberg.symmetry if config.eliashberg.symmetry else "random"} "
        f"for the {gamma_q_r_pp.channel.value}let channel."
    )

    einsum_str1 = "xyzabcdv,xyzdcv->xyzabv"
    path1 = np.einsum_path(einsum_str1, gchi0_q0_pp.mat, gap0, optimize=True)[1]
    einsum_str2 = "xyzadbcvp,xyzcdp->xyzabv"
    path2 = np.einsum_path(einsum_str2, gamma_x.mat, gap0, optimize=True)[1]

    norm = 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta

    def mv(gap: np.ndarray):
        gap_gg = np.fft.fftn(
            np.einsum(einsum_str1, gchi0_q0_pp.mat, gap.reshape(gap_shape), optimize=path1), axes=(0, 1, 2)
        )
        gap_gg_flipped = np.roll(np.flip(gap_gg, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        gap_new = np.einsum(einsum_str2, gamma_x.mat, gap_gg, optimize=path2) + sign * np.einsum(
            einsum_str2, gamma_x_flipped.mat, gap_gg_flipped, optimize=path2
        )
        return np.fft.ifftn(norm * gap_new, axes=(0, 1, 2)).flatten()

    mat = sp.sparse.linalg.LinearOperator(shape=(np.prod(gap_shape), np.prod(gap_shape)), matvec=mv)
    logger.log_info(
        f"Starting Lanczos method to retrieve largest{"" if config.eliashberg.n_eig > 1 else f" {config.eliashberg.n_eig}"} "
        f"eigenvalue{"" if config.eliashberg.n_eig == 1 else "s"} and eigenvector{"" if config.eliashberg.n_eig == 1 else "s"}."
    )

    lambda_max, gap_max = sp.sparse.linalg.eigsh(mat, k=1, tol=config.eliashberg.epsilon, v0=gap0, which="LA")
    logger.log_info("Finished Lanczos method for the largest eigenvalue.")
    logger.log_info(f"Maximum eigenvalue for {gamma_q_r_pp.channel.value}let channel is {lambda_max[0]:.6f}.")

    logger.log_info(
        f"Starting Lanczos method to retrieve the {config.eliashberg.n_eig} eigenvalue{"" if config.eliashberg.n_eig == 1 else "s"} "
        f"closest to one and their corresponding "
        f"eigenvector{"" if config.eliashberg.n_eig == 1 else "s"} using shift-invert mode."
    )
    lambdas, gaps = sp.sparse.linalg.eigsh(
        mat, k=config.eliashberg.n_eig, tol=config.eliashberg.epsilon, v0=gap0, sigma=1.0
    )
    logger.log_info(
        f"Finished Lanczos method for the {gamma_q_r_pp.channel.value}let channel for "
        f"the superconducting eigenvalues and gap functions."
    )
    idx = np.abs(lambdas - 1).argsort()
    lambdas = lambdas[idx]
    gaps = gaps[:, idx]
    logger.log_info(
        f"Superconducting eigenvalue{"" if config.eliashberg.n_eig == 1 else "s"} for the {gamma_q_r_pp.channel.value}let "
        f"channel are: {', '.join(f'{lam:.6f}' for lam in lambdas)}."
    )

    gaps = [
        GapFunction(gaps[..., i].reshape(gap_shape), gamma_q_r_pp.channel, gamma_q_r_pp.nq)
        for i in range(config.eliashberg.n_eig)
    ]

    lambdas = np.append(lambdas, lambda_max)
    gaps = np.append(
        gaps, [GapFunction(gap_max[..., 0].reshape(gap_shape), gamma_q_r_pp.channel, gamma_q_r_pp.nq)], axis=0
    )

    return lambdas, gaps


def create_local_reducible_pp_diagrams(giwk: GreensFunction, channel: SpinChannel) -> LocalFourPoint:
    """
    Create the reducible particle-particle diagrams for either singlet or triplet channels.
    """
    logger = config.logger

    if channel not in (SpinChannel.SING, SpinChannel.TRIP):
        raise ValueError("Channel must be either singlet or triplet.")

    giwk_loc = deepcopy(giwk)
    giwk_loc.mat = np.mean(giwk_loc.mat, axis=(0, 1, 2))
    gchi0 = BubbleGenerator.create_generalized_chi0(giwk_loc, config.box.niw_core, config.box.niv_core)
    del giwk_loc

    f_dens_loc = (
        LocalFourPoint.load(os.path.join(config.output.output_path, f"f_dens_loc.npy"), SpinChannel.DENS)
        .cut_niv(config.box.niv_core)  # f_dens is saved with an extended frequency box from asymptotics
        .change_frequency_notation_ph_to_pp()
    )
    f_magn_loc = (
        LocalFourPoint.load(os.path.join(config.output.output_path, f"f_magn_loc.npy"), SpinChannel.MAGN)
        .cut_niv(config.box.niv_core)  # f_magn is saved with an extended frequency box from asymptotics
        .change_frequency_notation_ph_to_pp()
    )
    logger.log_info("Loaded full local density and magnetic vertices and transformed them to pp notation.")

    f_r_loc = 0.5 * f_dens_loc + (-1.5 if channel == SpinChannel.SING else 0.5) * f_magn_loc
    del f_dens_loc, f_magn_loc
    f_r_loc.channel = channel
    logger.log_info(f"Constructed local {channel.value}let vertex.")

    gchi_dens_loc = LocalFourPoint.load(
        os.path.join(config.output.output_path, f"gchi_dens_loc.npy"), SpinChannel.DENS
    ).change_frequency_notation_ph_to_pp()
    gchi_magn_loc = LocalFourPoint.load(
        os.path.join(config.output.output_path, f"gchi_magn_loc.npy"), SpinChannel.MAGN
    ).change_frequency_notation_ph_to_pp()
    logger.log_info("Loaded local density and magnetic susceptibilities and transformed them to pp notation.")

    gchi_r_loc = 0.5 * gchi_dens_loc + (-1.5 if channel == SpinChannel.SING else 0.5) * gchi_magn_loc
    del gchi_dens_loc, gchi_magn_loc
    gchi_r_loc.channel = channel
    logger.log_info(f"Constructed local {channel.value}let susceptibility.")

    gchi0 = gchi0.change_frequency_notation_ph_to_pp()

    sign = 1 if channel == SpinChannel.SING else -1
    gamma_r_loc = config.sys.beta**2 * (4 * (gchi_r_loc - sign * gchi0).invert() + 2 * sign * gchi0.invert())
    logger.log_info(f"Constructed local {channel.value}let irreducible diagrams.")

    phi_r_loc = 1.0 / config.sys.beta**2 * (f_r_loc - gamma_r_loc)
    phi_r_loc.mat = phi_r_loc.mat[..., phi_r_loc.mat.shape[-3] // 2, :, :]  # we need w=0,v,v'
    phi_r_loc.update_original_shape()
    return phi_r_loc


def solve(giwk: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm):
    """
    Solve the Eliashberg equation for the superconducting eigenvalue and gap function.
    """
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    niv_pp = min(config.box.niw_core // 3, config.box.niv_core // 3)

    f_dens_pp = calculate_full_vertex_pp_w0(u_loc, v_nonloc, SpinChannel.DENS, niv_pp, mpi_dist_irrk)
    logger.log_info("Calculated full density vertex in pp notation.")
    f_magn_pp = calculate_full_vertex_pp_w0(u_loc, v_nonloc, SpinChannel.MAGN, niv_pp, mpi_dist_irrk)
    logger.log_info("Calculated full magnetic vertex in pp notation.")

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    mpi_dist_irrk.delete_file()

    gamma_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.log_info("Calculated full singlet pairing vertex in pp notation.")

    if config.eliashberg.save_pairing_vertex:
        gamma_sing_pp = gather_save_scatter(gamma_sing_pp, config.output.eliashberg_path, mpi_dist_irrk)
        config.logger.log_info(
            f"Saved {gamma_sing_pp.channel.value}let pairing vertex in pp notation in the irreducible BZ to file."
        )

    gamma_trip_pp = 0.5 * f_dens_pp + 0.5 * f_magn_pp
    gamma_trip_pp.channel = SpinChannel.TRIP
    del f_dens_pp, f_magn_pp
    logger.log_info("Calculated full triplet pairing vertex in pp notation.")

    if config.eliashberg.save_pairing_vertex:
        gamma_trip_pp = gather_save_scatter(gamma_trip_pp, config.output.eliashberg_path, mpi_dist_irrk)
        config.logger.log_info(
            f"Saved {gamma_trip_pp.channel.value}let pairing vertex in pp notation in the irreducible BZ to file."
        )

    gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat)
    gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat)

    if comm.rank == 0:
        gchi0_q0_pp = BubbleGenerator.create_generalized_chi0_pp_w0(giwk, niv_pp)
        logger.log_info("Created the bare bubble susceptibility in pp notation.")

        logger.log_info("Starting to calculate the local full UD vertex in pp notation.")
        f_ud_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_ud_loc.npy"))
        logger.log_info("Loaded the local full UD vertex from file.")
        f_ud_loc = transform_vertex_ph_to_pp_w0(f_ud_loc, niv_pp)
        logger.log_info(f"Calculated full local UD vertex in pp notation.")
        gamma_sing_pp -= f_ud_loc
        gamma_trip_pp -= f_ud_loc

        logger.log_info("Starting to calculate the local reducible singlet pairing diagrams.")
        phi_sing_loc = create_local_reducible_pp_diagrams(giwk, SpinChannel.SING)
        logger.log_info("Created the local reducible singlet pairing diagrams.")
        logger.log_info("Starting to calculate the local reducible triplet pairing diagrams.")
        phi_trip_loc = create_local_reducible_pp_diagrams(giwk, SpinChannel.TRIP)
        logger.log_info("Created the local reducible triplet pairing diagrams.")

        phi_ud_loc = 0.5 * (phi_sing_loc + phi_trip_loc)
        gamma_sing_pp -= phi_sing_loc
        gamma_trip_pp -= phi_trip_loc

        logger.log_info("Starting to solve the Eliashberg equation for the singlet channel.")
        lambdas_sing, gaps_sing = solve_eliashberg_lanczos(gamma_sing_pp, gchi0_q0_pp)
        logger.log_info("Finished solving the Eliashberg equation for the singlet channel.")
        logger.log_info("Starting to solve the Eliashberg equation for the triplet channel.")
        lambdas_trip, gaps_trip = solve_eliashberg_lanczos(gamma_trip_pp, gchi0_q0_pp)
        logger.log_info("Finished solving the Eliashberg equation for the triplet channel.")
    else:
        lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = (None,) * 4

    lambdas_sing = comm.bcast(lambdas_sing, root=0)
    lambdas_trip = comm.bcast(lambdas_trip, root=0)
    gaps_sing = comm.bcast(gaps_sing, root=0)
    gaps_trip = comm.bcast(gaps_trip, root=0)

    return lambdas_sing, lambdas_trip, gaps_sing, gaps_trip
