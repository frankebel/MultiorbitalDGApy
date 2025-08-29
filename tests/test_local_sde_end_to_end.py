from unittest.mock import patch

import mpi4py
import numpy as np
import pytest

import scdga.brillouin_zone as bz
import scdga.config as config
import scdga.dga_io
import scdga.greens_function
import scdga.hamiltonian
import scdga.local_four_point
import scdga.self_energy
import scdga.w2dyn_aux
from scdga import local_sde
from scdga.dga_logger import DgaLogger
from scdga.greens_function import GreensFunction
from scdga.local_four_point import LocalFourPoint
from scdga.n_point_base import SpinChannel


@pytest.fixture
def setup():
    folder = "./test_data/end_2_end"

    with patch("mpi4py.MPI.COMM_WORLD", wraps=mpi4py.MPI.COMM_WORLD) as comm_mock:
        config.logger = DgaLogger(comm_mock, "./")

        config.box.niw_core = -1
        config.box.niv_core = -1
        config.box.niv_shell = 10

        config.output.save_quantities = False
        config.output.do_plotting = False
        config.lattice.nk = (4, 4, 1)
        config.lattice.nq = config.lattice.nk
        config.lattice.k_grid = bz.KGrid(config.lattice.nk, symmetries=bz.two_dimensional_square_symmetries())
        config.lattice.q_grid = config.lattice.k_grid

        config.lattice.type = "from_wannierHK"
        config.lattice.interaction_type = "kanamori_from_dmft"

        config.lattice.er_input = f"{folder}/wannier.hk"
        config.dmft.input_path = folder
        config.dmft.do_sym_v_vp = True

        config.eliashberg.perform_eliashberg = False

        yield folder


@pytest.mark.parametrize(
    "niw_core, niv_core, niv_shell",
    [
        (10, 20, 0),
        (20, 10, 0),
        (20, 20, 0),
        (10, 20, 10),
        (20, 10, 10),
        (20, 20, 10),
        (-1, 20, 10),
        (20, -1, 10),
        (-1, -1, 10),
    ],
)
def test_extracts_dmft_quantities_correctly(setup, niw_core, niv_core, niv_shell):
    folder = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = scdga.dga_io.load_from_w2dyn_file_and_update_config()

    if niw_core == -1:
        assert config.box.niw_core == 20
    if niv_core == -1:
        assert config.box.niv_core == 20

    assert config.box.niv_full == config.box.niv_core + config.box.niv_shell

    g2_dens_ref = (
        LocalFourPoint.load(f"{folder}/g2_dens_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    )
    g2_magn_ref = (
        LocalFourPoint.load(f"{folder}/g2_magn_loc.npy")
        .to_full_niw_range()
        .cut_niw_and_niv(config.box.niw_core, config.box.niv_core)
    )

    g_dmft_ref_mat = np.load(f"{folder}/g_dmft.npy")
    s_dmft_ref_mat = np.load(f"{folder}/sigma_dmft.npy")

    assert np.allclose(g2_dens.mat, g2_dens_ref.mat)
    assert np.allclose(g2_magn.mat, g2_magn_ref.mat)

    assert np.allclose(g_dmft.mat, g_dmft_ref_mat)
    assert np.allclose(s_dmft.mat, s_dmft_ref_mat)

    assert config.box.niw_core == g2_dens.niw
    assert config.box.niv_core == g2_dens.niv

    u_loc = config.lattice.hamiltonian.get_local_u()
    assert u_loc.mat.shape == (2, 2, 2, 2)
    indices = np.arange(2)
    assert np.all(u_loc.mat[indices, indices, indices, indices] == 8)

    mask = np.ones(u_loc.mat.shape, dtype=bool)
    mask[indices, indices, indices, indices] = False
    assert np.all(u_loc.mat[mask] == 0)

    vq = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)
    assert vq.mat.shape == (4, 4, 1, 2, 2, 2, 2)
    assert np.allclose(vq.mat, np.zeros_like(vq.mat))


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10), (-1, 20, 10), (20, -1, 10), (-1, -1, 10)])
def test_calculates_local_sde_correctly(setup, niw_core, niv_core, niv_shell):
    folder = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = scdga.dga_io.load_from_w2dyn_file_and_update_config()

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = GreensFunction.create_g_loc(s_dmft.create_with_asympt_up_to_core(), ek)
    u_loc = config.lattice.hamiltonian.get_local_u()

    g_loc_ref_mat = np.load(f"{folder}/g_loc.npy")
    assert np.allclose(g_loc.mat, g_loc_ref_mat)

    (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, sigma_loc) = (
        local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)
    )

    def compare_quantity(obj_dens_sde, obj_magn_sde, dens_name: str, magn_name: str, num_vn_dimensions: int, niv: int):
        obj_dens_ref = LocalFourPoint.load(
            f"{folder}/{dens_name}.npy", SpinChannel.DENS, num_vn_dimensions=num_vn_dimensions
        )
        obj_magn_ref = LocalFourPoint.load(
            f"{folder}/{magn_name}.npy", SpinChannel.MAGN, num_vn_dimensions=num_vn_dimensions
        )

        assert np.allclose(obj_dens_sde.mat, obj_dens_ref.mat, atol=1e-3)
        assert np.allclose(obj_magn_sde.mat, obj_magn_ref.mat, atol=1e-3)

        assert obj_dens_sde.channel == obj_dens_ref.channel
        assert obj_magn_sde.channel == obj_magn_ref.channel

        assert obj_dens_sde.num_wn_dimensions == obj_dens_ref.num_wn_dimensions
        assert obj_magn_sde.num_wn_dimensions == obj_magn_ref.num_wn_dimensions

        assert obj_dens_sde.num_vn_dimensions == obj_dens_ref.num_vn_dimensions
        assert obj_magn_sde.num_vn_dimensions == obj_magn_ref.num_vn_dimensions

        assert obj_dens_sde.niw == obj_dens_ref.niw
        assert obj_magn_sde.niw == obj_magn_ref.niw

        assert obj_dens_sde.niv == obj_dens_ref.niv
        assert obj_magn_sde.niv == obj_magn_ref.niv

        assert obj_dens_sde.full_niw_range == obj_dens_ref.full_niw_range
        assert obj_magn_sde.full_niw_range == obj_magn_ref.full_niw_range

        assert obj_dens_sde.current_shape == (2, 2, 2, 2, 21) + (2 * niv,) * num_vn_dimensions
        assert obj_magn_sde.current_shape == (2, 2, 2, 2, 21) + (2 * niv,) * num_vn_dimensions

    compare_quantity(gamma_d, gamma_m, "gamma_dens_loc", "gamma_magn_loc", 2, config.box.niv_core)
    compare_quantity(chi_d, chi_m, "chi_dens_loc", "chi_magn_loc", 0, config.box.niv_core)
    compare_quantity(vrg_d, vrg_m, "vrg_dens_loc", "vrg_magn_loc", 1, config.box.niv_core)
    compare_quantity(f_d, f_m, "f_dens_loc", "f_magn_loc", 2, config.box.niv_full)
    compare_quantity(gchi_d, gchi_m, "gchi_dens_loc", "gchi_magn_loc", 2, config.box.niv_core)

    sigma_loc_ref = np.load(f"{folder}/siw_dga_local.npy")
    assert np.allclose(sigma_loc.mat, sigma_loc_ref, atol=1e-3)
