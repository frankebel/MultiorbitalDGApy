import argparse
import logging
import os

from ruamel.yaml import YAML

import config
from config import *
from mpi4py import MPI


class ConfigParser:
    def __init__(self):
        self._logger = logging.getLogger()
        self._config_file = None

    def parse_config(self, comm: MPI.Comm = None, path: str = "./", name: str = "dga_config.yaml"):
        """
        Parses the config file and builds the DgaConfig. It is then broadcasted to all processes.
        The config file location can be specified with the path and name arguments when executing the main python file.
        """
        parser = argparse.ArgumentParser(
            prog="DGApy", description="Multi-orbital dynamical vertex approximation solver."
        )
        parser.add_argument("-c", "--config", nargs="?", default=name, type=str, help=" Config file name. ")
        parser.add_argument("-p", "--path", nargs="?", default=path, type=str, help=" Path to the config file. ")

        if comm.rank == 0:
            args = parser.parse_args()
            self._config_file = YAML().load(open(os.path.join(args.path, args.config)))

        self._config_file = comm.bcast(self._config_file, root=0)
        self._build_config_from_file(self._config_file)
        return self._config_file

    def save_config_file(self, path: str = "./", name: str = "dga_config.yaml") -> None:
        with open(os.path.join(path, name), "w+") as file:
            YAML().dump(self._config_file, file)

    def _build_config_from_file(self, conf_file):
        config.box = self._build_box_config(conf_file)
        config.lattice = self._build_lattice_config(conf_file)
        config.dmft = self._build_dmft_config(conf_file)
        config.sys = self._build_system_config(conf_file)
        config.output = self._build_output_config(conf_file)

    def _build_box_config(self, config_file) -> BoxConfig:
        conf = BoxConfig()
        box_section = config_file["box_sizes"]

        conf.niw = int(box_section["niw"])
        conf.niv = int(box_section["niv"])
        conf.niv_shell = int(box_section["niv_shell"])

        return conf

    def _build_lattice_config(self, config_file) -> LatticeConfig:
        conf = LatticeConfig()
        lattice_section = config_file["lattice"]

        conf.nk = tuple(lattice_section["nk"])
        if "nq" not in lattice_section:
            self._logger.info("'nq' not set in config. Setting 'nq' = 'nk'.")
            conf.nq = conf.nk
        else:
            conf.nq = tuple(lattice_section["nq"])
        conf.symmetries = self._set_lattice_symmetries(lattice_section["symmetries"])
        conf.hamiltonian = self._set_hamiltonian(
            lattice_section["type"],
            lattice_section["hr_input"],
            lattice_section["interaction_type"],
            lattice_section["interaction_input"],
        )
        conf.k_grid = bz.KGrid(conf.nk, conf.symmetries)
        conf.q_grid = bz.KGrid(conf.nq, conf.symmetries)

        return conf

    def _set_lattice_symmetries(self, lattice_section) -> list[bz.KnownSymmetries]:
        if lattice_section == "two_dimensional_square":
            return bz.two_dimensional_square_symmetries()
        elif lattice_section == "quasi_one_dimensional_square":
            return bz.quasi_one_dimensional_square_symmetries()
        elif lattice_section == "simultaneous_x_y_inversion":
            return bz.simultaneous_x_y_inversion()
        elif lattice_section == "quasi_two_dimensional_square_symmetries":
            return bz.quasi_two_dimensional_square_symmetries()
        elif not lattice_section or lattice_section == "none":
            return []
        elif isinstance(lattice_section, (tuple, list)):
            symmetries = []
            for sym in lattice_section:
                if sym not in [s.value for s in bz.KnownSymmetries]:
                    raise NotImplementedError(f"Symmetry {sym} not supported.")
                symmetries.append(bz.KnownSymmetries(sym))
            return symmetries
        else:
            raise NotImplementedError(f"Symmetry {lattice_section} not supported.")

    def _set_hamiltonian(self, er_type: str, er_input: str | list, int_type: str, int_input: str | list) -> Hamiltonian:
        """
        Sets the Hamiltonian based on the input from the config file. \n
        The kinetic part can be set in two ways: \n
        1. By providing the single-band hopping parameters t, tp, tpp. \n
        2. By providing the path + filename to the wannier_hr file. \n
        The interaction can be set in three ways: \n
        1. By retrieving the data from the DMFT files. \n
        2. By providing the Kanamori interaction parameters [n_bands, U, J, (V)]. \n
        3. By providing the full path + filename to the U-matrix file. \n
        """
        ham = Hamiltonian()
        if er_type == "t_tp_tpp":
            if not isinstance(er_input, list):
                raise ValueError("Invalid input for t, tp, tpp.")
            ham = ham.kinetic_one_band_2d_t_tp_tpp(*er_input)
        elif er_type == "from_wannier90":
            if not isinstance(er_input, str):
                raise ValueError("Invalid input for wannier_hr.dat.")
            ham = ham.read_er_w2k(er_input)
        else:
            raise NotImplementedError(f"Hamiltonian type {er_type} not supported.")

        if int_type == "infer":
            # TODO: get stuff from dmft files somehow. Maybe put it somewhere else?
            # the data is already been read in the dga_io file
            pass
        elif int_type == "kanamori":
            if not isinstance(int_input, list) or not 3 <= len(int_input) <= 4:
                raise ValueError("Invalid input for kanamori interaction.")
            return ham.kanamori_interaction(*int_input)
        elif int_type == "custom":
            if not isinstance(int_input, str):
                raise ValueError("Invalid input for umatrix file.")
            return ham.read_umatrix(int_input)
        else:
            raise NotImplementedError(f"Interaction type {int_type} not supported.")

    def _build_dmft_config(self, config_file) -> DmftConfig:
        conf = DmftConfig()
        dmft_section = config_file["dmft_input"]

        conf.type = dmft_section["type"]
        conf.input_path = dmft_section["input_path"]
        conf.fname_1p = dmft_section["fname_1p"]
        conf.fname_2p = dmft_section["fname_2p"]
        conf.do_sym_v_vp = bool(dmft_section["do_sym_v_vp"])

        return conf

    def _build_system_config(self, config_file) -> SystemConfig:
        return SystemConfig()

    def _build_output_config(self, config_file) -> OutputConfig:
        conf = OutputConfig()
        output_section = config_file["output"]

        conf.do_plotting = bool(output_section["do_plotting"])
        conf.save_quantities = bool(output_section["save_quantities"])

        conf.output_path = output_section["output_path"]
        if not conf.output_path or conf.output_path == "":
            self._logger.info(f"'output_path' not set in config. Setting 'output_path' = '{config.dmft.input_path}'.")
            conf.output_path = config.dmft.input_path

        return conf
