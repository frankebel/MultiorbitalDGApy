import numpy as np

import config
from local_four_point import LocalFourPoint
from local_susceptibility import LocalSusceptibility
from interaction import LocalInteraction


class LocalIrreducibleVertex(LocalFourPoint):
    @staticmethod
    def create_irreducible_vertex(
        gchi_r: LocalSusceptibility, gchi0: LocalSusceptibility, u_loc: LocalInteraction
    ) -> "LocalIrreducibleVertex":
        # TODO: CHECK IF THIS WORKS
        chi_tilde_mat = u_loc.mat / (config.beta * config.beta) + (~gchi0).mat
        chi_tilde = LocalSusceptibility(chi_tilde_mat, gchi0.channel, 1, 1, gchi0.full_niw_range, gchi0.full_niv_range)
        chi_tilde_inv = chi_tilde.invert()

        gamma_r_mat = (~gchi_r).mat - chi_tilde_inv.mat + u_loc.mat / (config.beta * config.beta)

        return LocalIrreducibleVertex(gamma_r_mat, gchi_r.channel, 1, 2, gchi_r.full_niw_range, gchi_r.full_niv_range)
