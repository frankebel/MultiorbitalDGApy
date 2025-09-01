import logging
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

import scdga.brillouin_zone as bz
from scdga import config, dga_io, local_sde
from scdga import nonlocal_sde
from scdga.dga_logger import DgaLogger
from scdga.greens_function import GreensFunction


@pytest.fixture
def setup():
    folder = f"{os.path.dirname(os.path.abspath(__file__))}/test_data/end_2_end"

    comm_mock = MagicMock()
    comm_mock.Get_size.return_value = 1
    comm_mock.Get_rank.return_value = 0
    comm_mock.size = 1
    comm_mock.rank = 0
    comm_mock.barrier.return_value = None

    comm_mock.bcast.side_effect = lambda obj, root=0: obj

    def allreduce_mock(sendbuf, recvbuf, op=None):
        np.copyto(recvbuf, sendbuf)
        return recvbuf

    comm_mock.Allreduce.side_effect = allreduce_mock

    def gatherv_mock(sendbuf, recvlist, root=0):
        tot_result = recvlist[0]
        np.copyto(tot_result, sendbuf)
        return tot_result

    comm_mock.Gatherv.side_effect = gatherv_mock

    def scatterv_mock(sendlist, recvbuf, root=0):
        full_data = sendlist[0]
        np.copyto(recvbuf, full_data)
        return recvbuf

    comm_mock.Scatterv.side_effect = scatterv_mock

    with patch("mpi4py.MPI.COMM_WORLD", comm_mock):
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

        config.self_consistency.mixing = 1
        config.self_consistency.max_iter = 1

        yield folder, comm_mock


@pytest.fixture(autouse=True)
def mock_numpy_save(monkeypatch):
    """
    Automatically mock numpy.save for all tests in this file.
    """

    def fake_save(file, arr, **kwargs):
        pass

    monkeypatch.setattr(np, "save", fake_save)
    yield


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """
    Automatically mock logger.log for all tests in this file.
    """
    logger_mock = MagicMock()
    monkeypatch.setattr(logging, "getLogger", lambda name=None: logger_mock)
    monkeypatch.setattr(logging, "Logger", MagicMock(return_value=logger_mock))
    yield logger_mock


@pytest.mark.parametrize("niw_core, niv_core, niv_shell", [(20, 20, 10), (-1, 20, 10), (20, -1, 10), (-1, -1, 10)])
def test_calculates_nonlocal_sde_correctly(setup, niw_core, niv_core, niv_shell):
    folder, comm_mock = setup

    config.box.niw_core = niw_core
    config.box.niv_core = niv_core
    config.box.niv_shell = niv_shell

    g_dmft, s_dmft, g2_dens, g2_magn = dga_io.load_from_w2dyn_file_and_update_config()

    config.output.output_path = folder

    ek = config.lattice.hamiltonian.get_ek(config.lattice.k_grid)
    g_loc = GreensFunction.create_g_loc(s_dmft.create_with_asympt_up_to_core(), ek)

    u_loc = config.lattice.hamiltonian.get_local_u()
    v_nonloc = config.lattice.hamiltonian.get_vq(config.lattice.q_grid)

    (gamma_d, gamma_m, chi_d, chi_m, vrg_d, vrg_m, f_d, f_m, gchi_d, gchi_m, s_loc) = (
        local_sde.perform_local_schwinger_dyson(g_loc, g2_dens, g2_magn, u_loc)
    )

    sigma_dga_mat = (
        nonlocal_sde.calculate_self_energy_q(comm_mock, u_loc, v_nonloc, s_dmft, s_loc).decompress_q_dimension().mat
    )
    sigma_dga_ref = np.load(f"{folder}/sigma_dga.npy")

    assert np.allclose(sigma_dga_mat, sigma_dga_ref, atol=1e-5)
