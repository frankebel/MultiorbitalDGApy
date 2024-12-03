import numpy as np

import config
from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from local_susceptibility import LocalSusceptibility
from src.interaction import LocalInteraction
from src.local_irreducible_vertex import LocalIrreducibleVertex


def calculate_local_self_energy(
    g_loc: LocalGreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, u_loc: LocalInteraction
) -> LocalSelfEnergy:
    gchi_dens = LocalSusceptibility.create_generalized_chi(g2_dens, g_loc)
    gchi_magn = LocalSusceptibility.create_generalized_chi(g2_magn, g_loc)

    if config.do_plotting:
        gchi_dens.plot(omega=0, figure_name=f"Gchi_dens_w{0}")
        gchi_magn.plot(omega=0, figure_name=f"Gchi_magn_w{0}")

    gchi_0 = LocalSusceptibility.create_generalized_chi0(g_loc)

    gamma_dens = LocalIrreducibleVertex.create_irreducible_vertex(gchi_dens, gchi_0, u_loc)
    gamma_magn = LocalIrreducibleVertex.create_irreducible_vertex(gchi_magn, gchi_0, u_loc)

    return LocalSelfEnergy(np.array([]), g_loc.full_niv_range)
