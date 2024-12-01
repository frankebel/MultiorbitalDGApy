import numpy as np

from local_four_point import LocalFourPoint
from local_greens_function import LocalGreensFunction
from local_self_energy import LocalSelfEnergy
from local_susceptibility import LocalSusceptibility


def calculate_local_self_energy(
    giw: LocalGreensFunction, g2_dens: LocalFourPoint, g2_magn: LocalFourPoint
) -> LocalSelfEnergy:
    gchi_dens = LocalSusceptibility.create_from_g2(g2_dens, giw)
    gchi_magn = LocalSusceptibility.create_from_g2(g2_magn, giw)

    gchi_dens.plot(omega=0, figure_name=f"Gchi_dens_w{0}")
    gchi_magn.plot(omega=0, figure_name=f"Gchi_magn_w{0}")

    return LocalSelfEnergy(np.array([]), giw.full_niv_range)
