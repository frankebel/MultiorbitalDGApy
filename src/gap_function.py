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

    @property
    def n_bands(self) -> int:
        """
        Returns the number of bands.
        """
        return self.original_shape[1] if self.has_compressed_q_dimension else self.original_shape[3]

    def to_compound_indices(self):
        """
        Converts the indices of the gap function to compound indices.
        """
        if len(self.current_shape) == 2:  # [q,x]
            return self

        self.update_original_shape()
        self.mat = self.mat.reshape(self.nq_tot, self.n_bands**2 * 2 * self.niv)
        return self

    def to_full_indices(self, shape: tuple = None):
        """
        Converts an object stored with compound indices to an object that has unraveled orbital and frequency axes.
        """
        if len(self.current_shape) == (
            4 if self.has_compressed_q_dimension else 6
        ):  # [q,o1,o2,v] or [qx,qy,qz,o1,o2,v]
            return self

        self.original_shape = shape if shape is not None else self.original_shape

        if self.has_compressed_q_dimension:
            self.mat = self.mat.reshape(self.nq_tot, self.n_bands, self.n_bands, 2 * self.niv)
        else:
            self.mat = self.mat.reshape(*self.nq, self.n_bands, self.n_bands, 2 * self.niv)
        return self

    def add(self, other):
        """
        Adds two GapFunctions.
        """
        if not isinstance(other, GapFunction):
            raise TypeError("Can only add or subtract GapFunction objects.")

        self.compress_q_dimension()
        other = other.compress_q_dimension()

        return GapFunction(
            self.mat + other.mat,
            self.channel,
            nk=self.nq,
            full_niv_range=self.full_niv_range,
            has_compressed_q_dimension=True,
        )

    def sub(self, other):
        """
        Subtracts two GapFunctions.
        """
        return self.add(-other)

    def __add__(self, other):
        """
        Adds two GapFunctions.
        """
        return self.add(other)

    def __sub__(self, other):
        """
        Subtracts two GapFunctions.
        """
        return self.sub(other)

    def plot(
        self,
        kx: float,
        ky: float,
        name: str = "",
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
            plt.savefig(os.path.join(output_dir, f"GapFunction_{self.channel.value}let_{name}.png"))
        if show:
            plt.show()
        else:
            plt.close()
