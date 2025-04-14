import os

import mpi4py.MPI as MPI
import numpy as np
from numpy import ndarray
from scipy.sparse.linalg import LinearOperator, eigsh

import config
from four_point import FourPoint
from gap_function import GapFunction
from greens_function import GreensFunction
from interaction import LocalInteraction, Interaction
from local_four_point import LocalFourPoint
from matsubara_frequencies import MFHelper
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel, FrequencyNotation


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
    Gather the vertex function from all ranks, save it to file, and scatter it back to the original ranks.
    """
    f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)

    if mpi_dist_irrk.my_rank == 0:
        f_q_r.save(output_dir=file_path, name=f"f_{f_q_r.channel.value}_irrq")

    config.logger.log_info(
        f"Saved full {f_q_r.channel.value} vertex "
        f"{"in pp notation " if f_q_r.frequency_notation == FrequencyNotation.PP else ""}"
        f"(for the irreducible BZ) to file."
    )

    f_q_r.mat = mpi_dist_irrk.scatter(f_q_r.mat)
    return f_q_r


def create_gchi0_pp_w0(giwk: GreensFunction, niv_pp: int) -> FourPoint:
    r"""
    Returns the particle-particle bare bubble susceptibility from the Green's function. Returns the object with :math:`\omega = 0`.
    We have :math:`\chi_{0;abcd}^{\vec{k}(\omega=0)\nu} = -\beta * G_{ad}^k * G_{cb}^{-k}` with :math:`G_{cb}^{-k} = G_{bc}^{*k}`.
    """
    g = giwk.cut_niv(niv_pp).compress_q_dimension()

    eye_left = np.eye(config.sys.n_bands)[None, None, :, :, None, None]
    eye_right = np.eye(config.sys.n_bands)[None, :, None, None, :, None]

    g_left_mat = g.mat[:, :, None, None, :, :] * eye_left
    g_right_mat = np.conj(g.mat)[:, None, :, :, None, :] * eye_right
    gchi0_q = -config.sys.beta * g_left_mat * g_right_mat

    return FourPoint(gchi0_q, SpinChannel.NONE, config.lattice.nq, 0, 1, has_compressed_q_dimension=True)


def create_full_vertex_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, comm: MPI.Comm
) -> FourPoint:
    """
    Calculate the full vertex function in the given channel. See Eq. (3.140) and Eq. (3.141) in my thesis.
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
    logger.log_info(f"Full momentum-dependent ladder-vertex ({f_q_r.channel.value}) calculated.")
    logger.log_memory_usage(f"Full vertex ({f_q_r.channel.value})", f_q_r, comm.size)

    delete_files(
        config.output.eliashberg_path,
        f"vrg_q_{channel.value}_rank_{comm.rank}.npy",
        f"gchi_aux_q_{channel.value}_sum_rank_{comm.rank}.npy",
    )

    return f_q_r


def transform_vertex_ph_to_pp_w0(f_q_r: LocalFourPoint) -> LocalFourPoint | FourPoint:
    """
    Transform the vertex function from particle-hole notation to particle-particle notation. This is done by
    flipping the last Matsubara frequency in order to get v, -v' and then applying the necessary condition of w = v-v'.
    """
    is_local = not isinstance(f_q_r, FourPoint)

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]
    f_q_r_flip = f_q_r.cut_niv(niv_pp).to_full_niw_range().flip_frequency_axis(-1)
    del f_q_r
    f_q_r_pp_mat = np.zeros((*f_q_r_flip.current_shape[:-3], 2 * niv_pp, 2 * niv_pp), dtype=f_q_r_flip.mat.dtype)
    for idx, w in enumerate(MFHelper.wn(config.box.niw_core)):
        f_q_r_pp_mat[..., omega == w] = -f_q_r_flip[..., idx, omega == w]

    config.logger.log_info(
        f"Calculated full {f_q_r_flip.channel.value if f_q_r_flip.channel is not SpinChannel.NONE else "local UD"} "
        f"vertex in pp notation."
    )
    if is_local:
        return LocalFourPoint(f_q_r_pp_mat, f_q_r_flip.channel, 0, frequency_notation=FrequencyNotation.PP)
    return FourPoint(
        f_q_r_pp_mat, f_q_r_flip.channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP
    )


def calculate_full_vertex_pp_w0(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, mpi_dist_irrk: MpiDistributor
):
    """
    Calculates the full vertex function in PH notation and transforms it to PP notation.
    For the calculation of F, see Eq. (3.140) and Eq. (3.141) in my thesis.
    """
    group_size = max(mpi_dist_irrk.comm.size // 3, 1)
    color = mpi_dist_irrk.comm.rank // group_size
    sub_comm = mpi_dist_irrk.comm.Split(color, mpi_dist_irrk.comm.rank)

    f_q_r = None
    for i in range(sub_comm.size):
        if sub_comm.rank == i:
            f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, channel, mpi_dist_irrk.comm)
        sub_comm.Barrier()
    sub_comm.Free()

    if config.output.save_fq:
        f_q_r = gather_save_scatter(f_q_r, config.output.output_path, mpi_dist_irrk)

    return transform_vertex_ph_to_pp_w0(f_q_r)


def get_initial_gap_function(shape: tuple, channel: SpinChannel):
    if channel != SpinChannel.SING and channel != SpinChannel.TRIP:
        raise ValueError("Channel must be either SING or TRIP.")

    gap0 = np.zeros(shape, dtype=np.complex64)
    niv = shape[-1] // 2

    def d_wave(k_grid):
        return -np.cos(k_grid[0])[:, None, None] + np.cos(k_grid[1])[None, :, None]

    def p_wave_x(k_grid):
        return np.sin(k_grid[0])[:, None, None]

    def p_wave_y(k_grid):
        return np.sin(k_grid[1])[None, :, None]

    if config.eliashberg.symmetry == "d-wave":
        gap0[..., niv:] = np.repeat(d_wave(config.lattice.k_grid.grid)[:, :, :, None, None, None], niv, axis=-1)
    elif config.eliashberg.symmetry == "p-wave-x":
        gap0[..., niv:] = np.repeat(p_wave_x(config.lattice.k_grid.grid)[:, :, :, None, None, None], niv, axis=-1)
    elif config.eliashberg.symmetry == "p-wave-y":
        gap0[..., niv:] = np.repeat(p_wave_y(config.lattice.k_grid.grid)[:, :, :, None, None, None], niv, axis=-1)
    else:
        gap0 = np.random.random_sample(shape)

    v_sym = ""
    if config.eliashberg.symmetry == "d-wave":
        v_sym = "even" if channel == SpinChannel.SING else "odd"
    elif config.eliashberg.symmetry == "p-wave-x" or config.eliashberg.symmetry == "p-wave-y":
        v_sym = "odd" if channel == SpinChannel.SING else "even"

    if v_sym == "even":
        gap0[..., :niv] = gap0[..., niv:]
    elif v_sym == "odd":
        gap0[..., :niv] = -gap0[..., niv:]
    else:
        gap0 = np.random.random_sample(shape)

    return gap0


def solve_eliashberg_poweriter(gamma_q_r_pp: FourPoint, gchi0_q0_pp: FourPoint) -> tuple[list, list[GapFunction]]:
    """
    Solve the Eliashberg equation for the superconducting eigenvalue and gap function using the power iteration method.
    """
    gamma_q_r_pp = gamma_q_r_pp.map_to_full_bz(
        config.lattice.q_grid.irrk_inv, config.lattice.q_grid.nk
    ).decompress_q_dimension()

    gamma_x = gamma_q_r_pp.fft() if gamma_q_r_pp.channel == SpinChannel.SING else -gamma_q_r_pp.fft()
    gamma_x_flipped = gamma_x.flip_momentum_axis().flip_frequency_axis(-1)

    gap_shape = gamma_q_r_pp.nq + 2 * (gamma_q_r_pp.n_bands,) + (2 * gamma_q_r_pp.niv,)
    gchi0_q0_pp = gchi0_q0_pp.decompress_q_dimension()

    sign = 1 if gamma_q_r_pp.channel == SpinChannel.SING else -1

    gap0 = get_initial_gap_function(gap_shape, gamma_q_r_pp.channel)

    einsum_str1 = "xyzabcdv,xyzdcv->xyzabv"
    path1 = np.einsum_path(einsum_str1, gchi0_q0_pp.mat, gap0, optimize=True)[1]
    einsum_str2 = "xyzabcdvp,xyzdcp->xyzabv"
    path2 = np.einsum_path(einsum_str2, gamma_x.mat, gap0, optimize=True)[1]

    def mv(gap: np.ndarray):
        gap_gg = gap.reshape(gap_shape)
        gap_gg = np.fft.fftn(np.einsum(einsum_str1, gchi0_q0_pp.mat, gap_gg, optimize=path1))
        gap_gg_flipped = np.roll(np.flip(gap_gg, axis=(0, 1, 2)), shift=1, axis=(0, 1, 2))
        gap_new = np.einsum(einsum_str2, gamma_x.mat, gap_gg, optimize=path2)
        gap_new += sign * np.einsum(einsum_str2, gamma_x_flipped.mat, gap_gg_flipped, optimize=path2)
        gap_new *= 0.5 / config.lattice.q_grid.nk_tot / config.sys.beta
        return np.fft.ifftn(gap_new).flatten()

    mat = LinearOperator(shape=(np.prod(gap_shape), np.prod(gap_shape)), matvec=mv, dtype=np.complex64)
    lambdas, gaps = eigsh(mat, k=config.eliashberg.n_eig, which="LA", sigma=1, tol=config.eliashberg.epsilon, v0=gap0)
    idx = np.abs(lambdas - 1).argsort()
    lambdas = lambdas[idx]
    gaps = gaps[:, idx]

    gap_functions = []
    for i in range(config.eliashberg.n_eig):
        gap_functions.append(GapFunction(gaps[..., i], gamma_q_r_pp.channel, gamma_q_r_pp.nq))

    return lambdas, gap_functions


def solve_eliashberg_eig(gamma_q_r_pp: FourPoint, gchi0_q0_pp: FourPoint) -> tuple[float, GapFunction]:
    r"""
    Solve the Eliashberg equation for the superconducting eigenvalue and gap function. For this we have to solve the
    eigenvalue problem for the matrix :math:`\mp\Gamma_{s/t;1b2a}^{k-k'\nu\nu'} @ \chi_{0}^{pp;k-k'\nu'\nu''}_{abcd}`.
    We do this by taking the eigenvalues of the matrix in real space and then transforming back to momentum space. To
    be more precise, we compute the discrete inverse Fourier transform of both the irreducible vertex and the bare
    bubble susceptibility in pp-notation. We then multiply the two matrices in real space and take the leading eigenvalue
    and corresponding eigenfunction of the resulting matrix. We then forward Fourier transform the eigenfunction to get
    the gap function in momentum space.
    """
    logger = config.logger

    gamma_q_r_pp = gamma_q_r_pp.map_to_full_bz(config.lattice.q_grid.irrk_inv, config.lattice.q_grid.nk)
    logger.log_info(f"Mapped Gamma_pp ({gamma_q_r_pp.channel.value}let) to full BZ.")

    factor = 1 if gamma_q_r_pp.channel == SpinChannel.SING else -1
    gamma = gamma_q_r_pp.ifft().times("kibjavp,kabcdp->kijcdvp", gchi0_q0_pp.ifft())
    gamma += factor * gamma_q_r_pp.flip_momentum_axis().ifft().flip_frequency_axis(-1).times(
        "kibjavp,kabcdp->kijcdvp", gchi0_q0_pp.flip_momentum_axis().ifft()
    )
    gamma *= 0.5 / config.sys.beta

    obj = FourPoint(gamma, gamma_q_r_pp.channel, config.lattice.nk, 0, 2, has_compressed_q_dimension=True)
    logger.log_info("Calculated the matrix for the eigenvalue problem.")
    eigvals, eigvecs = obj.find_eigendecomposition()
    logger.log_info("Calculated the eigenvalues and eigenvectors of the matrix.")

    lam_r = eigvals.real.max()
    logger.log_info(f"Found the largest eigenvalue for {gamma_q_r_pp.channel.value}let channel: {lam_r:.6f}.")

    indices = np.argmax(eigvals.real, axis=-1)[..., None, None]
    gap_r = np.take_along_axis(eigvecs, indices, axis=-2).squeeze(-2)
    gap_r = gap_r.reshape(config.lattice.k_grid.nk_tot, config.sys.n_bands, config.sys.n_bands, 2 * gamma_q_r_pp.niv)
    gap_r = GapFunction(gap_r, gamma_q_r_pp.channel, gamma_q_r_pp.nq, True, True).ifft()

    logger.log_info(f"Found the the {gamma_q_r_pp.channel.value}let gap function.")
    logger.log_info(f"Eliashberg equation for {gamma_q_r_pp.channel.value}let channel solved.")
    return lam_r, gap_r


def solve(giwk: GreensFunction, u_loc: LocalInteraction, v_nonloc: Interaction, comm: MPI.Comm):
    """
    Solve the Eliashberg equation for the superconducting eigenvalue and gap function.
    """
    logger = config.logger

    mpi_dist_irrk = MpiDistributor.create_distributor(ntasks=config.lattice.q_grid.nk_irr, comm=comm, name="Q")
    irrk_q_list = config.lattice.q_grid.get_irrq_list()
    my_irr_q_list = irrk_q_list[mpi_dist_irrk.my_slice]

    v_nonloc = v_nonloc.reduce_q(my_irr_q_list)

    f_dens_pp = calculate_full_vertex_pp_w0(u_loc, v_nonloc, SpinChannel.DENS, mpi_dist_irrk)
    f_magn_pp = calculate_full_vertex_pp_w0(u_loc, v_nonloc, SpinChannel.MAGN, mpi_dist_irrk)
    logger.log_info("Created full density and magnetic vertex in pp notation.")

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    mpi_dist_irrk.delete_file()

    gamma_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    gamma_sing_pp.channel = SpinChannel.SING
    logger.log_info("Created full singlet pairing vertex in pp notation.")

    if config.eliashberg.save_pairing_vertex:
        gamma_sing_pp = gather_save_scatter(gamma_sing_pp, config.output.eliashberg_path, mpi_dist_irrk)

    gamma_trip_pp = 0.5 * f_dens_pp + 0.5 * f_magn_pp
    gamma_trip_pp.channel = SpinChannel.TRIP
    del f_dens_pp, f_magn_pp
    logger.log_info("Created full triplet pairing vertex in pp notation.")

    if config.eliashberg.save_pairing_vertex:
        gamma_trip_pp = gather_save_scatter(gamma_trip_pp, config.output.eliashberg_path, mpi_dist_irrk)

    gamma_sing_pp.mat = mpi_dist_irrk.gather(gamma_sing_pp.mat)
    gamma_trip_pp.mat = mpi_dist_irrk.gather(gamma_trip_pp.mat)

    if comm.rank == 0:
        niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
        gchi0_q0_pp = create_gchi0_pp_w0(giwk, niv_pp)
        logger.log_info("Created the bare bubble susceptibility in pp notation and for w = 0.")

        f_ud_loc = LocalFourPoint.load(os.path.join(config.output.output_path, f"f_ud_loc.npy"))
        logger.log_info("Loaded the local full UD vertex from file.")
        f_ud_loc = transform_vertex_ph_to_pp_w0(f_ud_loc)
        # gamma_sing_pp -= f_ud_loc
        # gamma_trip_pp -= f_ud_loc

        lambdas_sing, gaps_sing = solve_eliashberg_poweriter(gamma_sing_pp, gchi0_q0_pp)
        lambdas_trip, gaps_trip = solve_eliashberg_poweriter(gamma_trip_pp, gchi0_q0_pp)
    else:
        lambdas_sing, lambdas_trip, gaps_sing, gaps_trip = (None,) * 4

    lambdas_sing = comm.bcast(lambdas_sing, root=0)
    lambdas_trip = comm.bcast(lambdas_trip, root=0)
    gaps_sing = comm.bcast(gaps_sing, root=0)
    gaps_trip = comm.bcast(gaps_trip, root=0)

    return lambdas_sing, lambdas_trip, gaps_sing, gaps_trip
