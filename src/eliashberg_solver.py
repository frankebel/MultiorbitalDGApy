import os

import mpi4py.MPI as MPI
import numpy as np

import config
from four_point import FourPoint
from greens_function import GreensFunction
from interaction import LocalInteraction, Interaction
from matsubara_frequencies import MFHelper
from mpi_distributor import MpiDistributor
from n_point_base import SpinChannel, FrequencyNotation
from gap_function import GapFunction


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


def gather_save_scatter(
    f_q_r: FourPoint, file_path: str, mpi_dist_irrk: MpiDistributor, scatter: bool = True
) -> FourPoint:
    """
    Gather the full vertex function from all ranks, save it to file, and scatter it back to the original rank.
    """
    if not config.output.save_quantities:
        return f_q_r

    f_q_r.mat = mpi_dist_irrk.gather(f_q_r.mat)
    if mpi_dist_irrk.my_rank == 0:
        f_q_r.save(output_dir=file_path, name=f"f_{f_q_r.channel.value}_irrq")
        config.logger.log_info(
            f"Saved full {f_q_r.channel.value}let vertex "
            f"{"in pp notation " if f_q_r.frequency_notation == FrequencyNotation.PP else ""}"
            f"(for the irreducible BZ) to file."
        )
    if scatter:
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


def create_full_vertex_pp_q_r(
    u_loc: LocalInteraction, v_nonloc: Interaction, channel: SpinChannel, mpi_dist_irrk: MpiDistributor
) -> FourPoint:
    """
    Calculate the full vertex in pp notation. For details, see the supplementary information of
    Phys. Rev. B 99, 041115(R) (2019).
    """
    f_q_r = create_full_vertex_q_r(u_loc, v_nonloc, channel, mpi_dist_irrk.comm)
    f_q_r = gather_save_scatter(f_q_r, config.output.eliashberg_path, mpi_dist_irrk, scatter=True)

    niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
    vn = MFHelper.vn(niv_pp)
    omega = vn[:, None] - vn[None, :]

    f_q_r_flip = f_q_r.cut_niv(niv_pp).to_full_niw_range().flip_axis(-1)
    del f_q_r

    f_q_r_pp_mat = np.zeros((*f_q_r_flip.current_shape[:5], 2 * niv_pp, 2 * niv_pp), dtype=f_q_r_flip.mat.dtype)
    for idx, w in enumerate(MFHelper.wn(config.box.niw_core)):
        f_q_r_pp_mat[..., omega == w] = -f_q_r_flip[..., idx, omega == w]
    del f_q_r_flip

    config.logger.log_info(f"Calculated full {channel.value} vertex in pp notation.")
    return FourPoint(f_q_r_pp_mat, channel, config.lattice.q_grid.nk, 0, 2, True, True, True, FrequencyNotation.PP)


def solve_eliashberg_eig(gamma_q_r_pp: FourPoint, gchi0_q0_pp_ifft: FourPoint) -> tuple[float, GapFunction]:
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

    factor = -1 if gamma_q_r_pp.channel == SpinChannel.SING else 1
    mat = factor * 0.5 / config.sys.beta * gamma_q_r_pp.ifft().times("kibjavp,kabcdp->kijcdvp", gchi0_q0_pp_ifft)
    logger.log_info("Calculated the matrix for the eigenvalue problem.")
    eigvals, eigvecs = np.linalg.eig(mat)
    logger.log_info("Calculated the eigenvalues and eigenvectors of the matrix.")

    lam_r = eigvals.real.max()
    logger.log_info(f"Found the largest eigenvalue for {gamma_q_r_pp.channel.value}let channel: {lam_r:.6f}.")

    indices = np.argmax(eigvals.real, axis=-1)[..., None, None]
    gap_r = np.take_along_axis(eigvecs, indices, axis=-2).squeeze(-2)
    gap_r = GapFunction(gap_r, gamma_q_r_pp.channel, gamma_q_r_pp.nq, True).fft()

    logger.log_info(f"Found the the {gamma_q_r_pp.channel.value}let gap function.")
    logger.log_info(f"Eliashberg equation for ({gamma_q_r_pp.channel.value}) channel solved.")
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

    # TODO: These objects are very large! Use Subcommunicator to reduce memory usage.

    f_dens_pp = create_full_vertex_pp_q_r(u_loc, v_nonloc, SpinChannel.DENS, mpi_dist_irrk)
    f_magn_pp = create_full_vertex_pp_q_r(u_loc, v_nonloc, SpinChannel.MAGN, mpi_dist_irrk)
    logger.log_info("Created full density and magnetic pairing vertex in pp notation.")

    delete_files(config.output.eliashberg_path, f"gchi0_q_inv_rank_{comm.rank}.npy")
    mpi_dist_irrk.delete_file()

    f_sing_pp = 0.5 * f_dens_pp - 1.5 * f_magn_pp
    f_sing_pp.channel = SpinChannel.SING
    logger.log_info("Created full singlet pairing vertex in pp notation.")

    f_sing_pp = gather_save_scatter(f_sing_pp, config.output.eliashberg_path, mpi_dist_irrk, scatter=False)

    f_trip_pp = 0.5 * f_dens_pp + 0.5 * f_magn_pp
    f_trip_pp.channel = SpinChannel.TRIP
    del f_dens_pp, f_magn_pp
    logger.log_info("Created full triplet pairing vertex in pp notation.")

    f_trip_pp = gather_save_scatter(f_trip_pp, config.output.eliashberg_path, mpi_dist_irrk, scatter=False)

    if comm.rank == 0:
        niv_pp = min(config.box.niw_core // 2, config.box.niv_core // 2)
        gchi0_q0_pp_ifft = create_gchi0_pp_w0(giwk, niv_pp).ifft()
        logger.log_info("Created the bare bubble susceptibility in pp notation and for w = 0.")

        lam_sing, gap_sing = solve_eliashberg_eig(f_sing_pp, gchi0_q0_pp_ifft)
        lam_trip, gap_trip = solve_eliashberg_eig(f_trip_pp, gchi0_q0_pp_ifft)
    else:
        lam_sing, lam_trip, gap_sing, gap_trip = (None,) * 4

    lam_sing = comm.bcast(lam_sing, root=0)
    lam_trip = comm.bcast(lam_trip, root=0)
    gap_sing = comm.bcast(gap_sing, root=0)
    gap_trip = comm.bcast(gap_trip, root=0)

    return lam_sing, lam_trip, gap_sing, gap_trip
