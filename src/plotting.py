import numpy as np
from matplotlib import pyplot as plt

import config

from matsubara_frequencies import MFHelper
from local_greens_function import LocalGreensFunction


def sigma_loc_checks(
    siw_arr: list[np.ndarray],
    labels: list[str],
    beta: float,
    output_dir: str = "./",
    show: bool = False,
    save: bool = True,
    name: str = "",
    xmax: float = 0,
) -> None:
    """
    siw_arr: list of local self-energies for routine plots.
    """
    if xmax == 0:
        xmax = 5 + 2 * beta
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
    axes = axes.flatten()

    for i, siw in enumerate(siw_arr):
        vn = MFHelper.vn(np.size(siw) // 2)
        axes[0].plot(vn, siw.real, label=labels[i])
    axes[0].set_ylabel(r"$\Re \Sigma(i\nu_n)$")
    axes[0].set_xlabel(r"$\nu_n$")

    for i, siw in enumerate(siw_arr):
        vn = MFHelper.vn(np.size(siw) // 2)
        axes[1].plot(vn, siw.imag, label=labels[i])
    axes[1].set_ylabel(r"$\Im \Sigma(i\nu_n)$")
    axes[1].set_xlabel(r"$\nu_n$")

    for i, siw in enumerate(siw_arr):
        vn = MFHelper.vn(np.size(siw) // 2)
        axes[2].loglog(vn, siw.real, label=labels[i])
    axes[2].set_ylabel(r"$\Re \Sigma(i\nu_n)$")
    axes[2].set_xlabel(r"$\nu_n$")

    for i, siw in enumerate(siw_arr):
        vn = MFHelper.vn(np.size(siw) // 2)
        axes[3].loglog(vn, np.abs(siw.imag), label=labels[i])
    axes[3].set_ylabel(r"$|\Im \Sigma(i\nu_n)|$")
    axes[3].set_xlabel(r"$\nu_n$")

    axes[0].set_xlim(0, xmax)
    axes[1].set_xlim(0, xmax)
    axes[2].set_xlim(None, xmax)
    axes[3].set_xlim(None, xmax)
    plt.legend()
    axes[1].set_ylim(None, 0)
    plt.tight_layout()
    if save:
        plt.savefig(output_dir + f"/sde_" + name + "_check.png")
    if show:
        plt.show()
    else:
        plt.close()


def chi_checks(
    chi_dens: list[np.ndarray],
    chi_magn: list[np.ndarray],
    labels: list[str],
    g_loc: LocalGreensFunction,
    output_dir: str = "./",
    orbs=[0, 0],
    show: bool = False,
    save: bool = True,
    name: str = "",
):
    """
    Routine plots to inspect chi_dens and chi_magn
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 5), dpi=500)
    axes = axes.flatten()
    niw_chi_input = np.size(chi_dens[0][*orbs, :])

    for i, cd in enumerate(chi_dens):
        axes[0].plot(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i])
    axes[0].set_ylabel(r"$\Re \chi(i\omega_n)_{dens}$")
    axes[0].legend()

    for i, cd in enumerate(chi_magn):
        axes[1].plot(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i])
    axes[1].set_ylabel(r"$\Re \chi(i\omega_n)_{magn}$")
    axes[1].legend()

    for i, cd in enumerate(chi_dens):
        axes[2].loglog(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i], ms=0)
    axes[2].loglog(
        MFHelper.wn(niw_chi_input),
        np.real(1 / (1j * MFHelper.wn(niw_chi_input, config.sys.beta) + 0.000001) ** 2 * g_loc.e_kin) * 2,
        ls="--",
        label="Asympt",
        ms=0,
    )
    axes[2].set_ylabel(r"$\Re \chi(i\omega_n)_{dens}$")
    axes[2].legend()

    for i, cd in enumerate(chi_magn):
        axes[3].loglog(MFHelper.wn(len(cd[*orbs, :]) // 2), cd[*orbs, :].real, label=labels[i], ms=0)
    axes[3].loglog(
        MFHelper.wn(niw_chi_input),
        np.real(1 / (1j * MFHelper.wn(niw_chi_input, config.sys.beta) + 0.000001) ** 2 * g_loc.e_kin) * 2,
        "--",
        label="Asympt",
        ms=0,
    )
    axes[3].set_ylabel(r"$\Re \chi(i\omega_n)_{magn}$")
    axes[3].legend()
    axes[0].set_xlim(-1, 10)
    axes[1].set_xlim(-1, 10)
    plt.tight_layout()
    if save:
        plt.savefig(output_dir + f"/chi_dens_magn_" + name + ".png")
    if show:
        plt.show()
    else:
        plt.close()
