import config
from local_greens_function import LocalGreensFunction
from matsubara_frequency_helper import MFHelper
from local_four_point import LocalFourPoint
from local_n_point import Channel


class LocalSusceptibility(LocalFourPoint):
    @staticmethod
    def create_from_g2(g2: LocalFourPoint, giw: LocalGreensFunction) -> "LocalSusceptibility":
        """gchi_r:1/eV^3 = beta:1/eV * (G2_r:1/eV^2 - 2 * GG:1/eV^2 delta_dens delta_w0)"""
        mat = config.beta * g2.mat

        if g2.channel == Channel.DENS:
            ggv = LocalSusceptibility._get_ggv_mat(giw, niv_slice=g2.niv)
            wn = MFHelper.get_wn_int(g2.niw)
            mat[:, :, :, :, wn == 0] = config.beta * (g2.mat[:, :, :, :, wn == 0] - 2.0 * ggv.mat)

        return LocalSusceptibility(mat, g2.channel, 1, 1, g2.full_niw_range, g2.full_niv_range)

    @staticmethod
    def _get_ggv_mat(giw: LocalGreensFunction, niv_slice: int = -1) -> LocalFourPoint:
        if niv_slice == -1:
            niv_slice = giw.niv
        copy = giw
        copy.mat = copy.mat[..., copy.niv - niv_slice : copy.niv + niv_slice]
        return copy * copy
