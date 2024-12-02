import numpy as np

import config
from local_four_point import LocalFourPoint
from local_susceptibility import LocalSusceptibility


class LocalIrreducibleVertex(LocalFourPoint):
    @staticmethod
    def create_irreducible_vertex(
        gchi_r: LocalSusceptibility, gchi0: LocalSusceptibility, ur_local: np.ndarray
    ) -> "LocalIrreducibleVertex":
        chi_tilde_inv = (
            ur_local * np.eye(gchi0.niv)[None, None, None, None, :, :] / (config.beta * config.beta) + (~gchi0).mat
        )
