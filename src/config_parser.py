import argparse
import os

from mpi4py import MPI
from ruamel.yaml import YAML

import config
from config import *
from dga_logger import DgaLogger


class ConfigParser:
    def __init__(self):
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

        config.logger = DgaLogger(comm)

        self._config_file = comm.bcast(self._config_file, root=0)
        config.current_rank = comm.rank

        self._build_config_from_file(self._config_file)

    def save_config_file(self, path: str = "./", name: str = "dga_config.yaml") -> None:
        with open(os.path.join(path, name), "w+") as file:
            YAML().dump(self._config_file, file)

    def _build_config_from_file(self, conf_file):
        config.dmft = self._build_dmft_config(conf_file)
        config.output = self._build_output_config(conf_file)
        config.box = self._build_box_config(conf_file)
        config.lattice = self._build_lattice_config(conf_file)
        config.sys = self._build_system_config(conf_file)

    def _build_box_config(self, config_file) -> BoxConfig:
        conf = BoxConfig()
        box_section = config_file["box_sizes"]

        conf.niw = int(box_section["niw"])
        conf.niv = int(box_section["niv"])
        conf.niv_asympt = int(box_section["niv_asympt"])
        if conf.niv_asympt <= 0:
            config.logger.log_info(f"'niv_asympt' is set to {conf.niv_asympt}. No asymptotics will be used.")
            conf.niv_asympt = 0
        conf.niv_full = conf.niv + conf.niv_asympt

        return conf

    def _build_lattice_config(self, config_file) -> LatticeConfig:
        conf = LatticeConfig()
        lattice_section = config_file["lattice"]

        conf.nk = tuple[int, int, int](lattice_section["nk"])
        if "nq" not in lattice_section:
            config.logger.log_info("'nq' not set in config. Setting 'nq' = 'nk'.")
            conf.nq = conf.nk
        else:
            conf.nq = tuple[int, int, int](lattice_section["nq"])
        conf.symmetries = self._set_lattice_symmetries(lattice_section["symmetries"])
        conf.type = lattice_section["type"]
        conf.er_input = lattice_section["hr_input"]
        conf.interaction_type = lattice_section["interaction_type"]
        conf.interaction_input = lattice_section["interaction_input"]
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

    def _build_dmft_config(self, config_file) -> DmftConfig:
        conf = DmftConfig()
        dmft_section = config_file["dmft_input"]

        conf.type = dmft_section["type"]
        conf.input_path = dmft_section["input_path"]
        conf.fname_1p = dmft_section["fname_1p"]
        conf.fname_2p = dmft_section["fname_2p"]
        conf.do_sym_v_vp = bool(dmft_section["do_sym_v_vp"])

        return conf

    def _build_system_config(self, _) -> SystemConfig:
        return SystemConfig()

    def _build_output_config(self, config_file) -> OutputConfig:
        conf = OutputConfig()
        output_section = config_file["output"]

        conf.do_plotting = bool(output_section["do_plotting"])
        conf.save_quantities = bool(output_section["save_quantities"])

        conf.output_path = output_section["output_path"]
        if not conf.output_path or conf.output_path == "":
            config.logger.log_info(
                f"'output_path' not set in config. Setting 'output_path' = '{config.dmft.input_path}'."
            )
            conf.output_path = config.dmft.input_path

        return conf
