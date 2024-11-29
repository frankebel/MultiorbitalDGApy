import config
import dga_io
from hamiltonian import Hamiltonian, HamiltonianBuilder
import logging


logging.basicConfig(level=logging.DEBUG)

dmft_input = dga_io.load_from_w2dyn_file(config.input_path, config.dmft_1p_filename, config.dmft_2p_filename)
hamiltonian: Hamiltonian = (
    HamiltonianBuilder()
    .add_kinetic_one_band_2d_t_tp_tpp(*config.lattice_er_input)
    .add_single_band_interaction(dmft_input['u'])
)

g2_dens = dmft_input['g4iw_dens']
g2_magn = dmft_input['g4iw_magn']

dga_io.update_frequency_boxes(g2_dens.niv, g2_dens.niw)

g2_dens = g2_dens.cut_niw_and_niv(config.niw, config.niv)
g2_magn = g2_magn.cut_niw_and_niv(config.niw, config.niv)
if config.do_sym_v_vp:
    g2_dens = g2_dens.symmetrize_v_vp()
    g2_magn = g2_magn.symmetrize_v_vp()


print(dmft_input)
