import numpy as np
import scdga.w2dyn_aux
import scdga.self_energy
import scdga.config as config
import scdga.greens_function
import scdga.local_four_point
import scdga.hamiltonian
import scdga.brillouin_zone
import scdga.dga_io


def test_calculates_local_quantities_correctly():
    config.output.save_quantities = False
    config.output.do_plotting = False

    config.dmft.input_path = "../test_data"

    # scdga.dga_io.load_from_w2dyn_file_and_update_config()

    pass
