import os

import matplotlib.pyplot as plt
import numpy as np

from local_n_point import LocalNPoint
from n_point_base import IAmNonLocal, IHaveChannel, SpinChannel, FrequencyNotation
import plotting


class GapFunction(LocalNPoint, IAmNonLocal, IHaveChannel):
    """
    Represents the superconducting gap function.
    """

    def __init__(
        self,
        mat: np.ndarray,
        channel: SpinChannel = SpinChannel.NONE,
        nk: tuple[int, int, int] = (1, 1, 1),
        full_niv_range: bool = True,
        has_compressed_q_dimension: bool = False,
    ):
        LocalNPoint.__init__(self, mat, 2, 0, 1, full_niv_range=full_niv_range)
        IAmNonLocal.__init__(self, mat, nk, has_compressed_q_dimension=has_compressed_q_dimension)
        IHaveChannel.__init__(self, channel, FrequencyNotation.PP)

    def plot(
        self,
        kx: float,
        ky: float,
        orbs: np.ndarray | list | tuple = (0, 0),
        do_save: bool = True,
        output_dir="./",
        cmap="RdBu",
        scatter=None,
        show: bool = False,
    ):
        if len(orbs) != 2:
            raise ValueError("'orbs' needs to be of size 2.")

        gap_func_shifted = self.shift_k_by_pi()
        niv_pp = gap_func_shifted.shape[-1] // 2
        gap_func_shifted = gap_func_shifted[:, :, 0, orbs[0], orbs[1], niv_pp - 1 : niv_pp + 1]
        fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=500)
        axes = axes.flatten()
        im1 = axes[0].pcolormesh(kx, ky, gap_func_shifted[..., 0].T.real, cmap=cmap)
        im2 = axes[1].pcolormesh(kx, ky, gap_func_shifted[..., 1].T.real, cmap=cmap)
        axes[0].set_title(r"$\nu_{n=0}$")
        axes[1].set_title(r"$\nu_{n=-1}$")
        for ax in axes:
            ax.set_xlabel(r"$k_x$")
            ax.set_ylabel(r"$k_y$")
            ax.set_aspect("equal")
            plotting.add_afzb(ax=ax, kx=kx, ky=ky, lw=1.0, marker="")
        fig.suptitle(f"{self.channel.value}let")
        fig.colorbar(im1, ax=(axes[0]), aspect=15, fraction=0.08, location="right", pad=0.05)
        fig.colorbar(im2, ax=(axes[1]), aspect=15, fraction=0.08, location="right", pad=0.05)
        if scatter is not None:
            for ax in axes:
                colours = plt.cm.get_cmap(cmap)(np.linspace(0, 1, np.shape(scatter)[0]))
                ax.scatter(scatter[:, 0], scatter[:, 1], marker="o", c=colours)
        plt.tight_layout()
        if do_save:
            plt.savefig(os.path.join(output_dir, f"GapFunction_{self.channel.value}let.png"))
        if show:
            plt.show()
        else:
            plt.close()
