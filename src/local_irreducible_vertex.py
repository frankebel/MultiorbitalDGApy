import config
from interaction import LocalInteraction
from local_four_point import LocalFourPoint
from local_susceptibility import LocalSusceptibility


class LocalIrreducibleVertex(LocalFourPoint):
    @staticmethod
    def create_irreducible_vertex(
        gchi_r: LocalSusceptibility, gchi0: LocalSusceptibility, u_loc: LocalInteraction
    ) -> "LocalIrreducibleVertex":
        chi_tilde_mat = ((~gchi0) + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)).mat
        chi_tilde = LocalSusceptibility(
            chi_tilde_mat,
            gchi_r.channel,
            full_niw_range=gchi0.full_niw_range,
            full_niv_range=gchi0.full_niv_range,
        )

        gammar_mat = ((~gchi_r) - chi_tilde + u_loc.as_channel(gchi_r.channel) / (config.beta * config.beta)).mat
        return LocalIrreducibleVertex(
            gammar_mat,
            gchi_r.channel,
            full_niw_range=gchi_r.full_niw_range,
            full_niv_range=gchi_r.full_niv_range,
        )
