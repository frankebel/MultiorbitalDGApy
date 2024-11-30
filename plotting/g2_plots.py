import numpy as np
from src.local_four_point import LocalFourPoint


def default_g2_plots(g2_dens: LocalFourPoint, g2_magn: LocalFourPoint, output_dir):
    """Default plots for the two-particle Green's function"""
    g2_dens.plot(0, pdir=output_dir, name="G2_dens")
    g2_magn.plot(0, pdir=output_dir, name="G2_magn")
    g2_magn.plot(10, pdir=output_dir, name="G2_magn")
    g2_magn.plot(-10, pdir=output_dir, name="G2_magn")
