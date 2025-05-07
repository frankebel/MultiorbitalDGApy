import numpy as np

from scdga import config
from scdga.four_point import FourPoint
from scdga.local_four_point import LocalFourPoint


def get_lambda_start(chi_r: np.ndarray) -> float:
    """
    Returns the starting value for the lambda correction.
    """
    w0 = chi_r.shape[-1] // 2
    return -np.min(1.0 / chi_r[..., w0].real)


def apply_lambda(chi_r: np.ndarray, lambda_: float) -> np.ndarray:
    """
    Applies the lambda correction to the physical susceptibility.
    """
    return 1.0 / (1.0 / chi_r + lambda_)


def find_lambda(
    chi_r_mat: np.ndarray,
    chi_r_loc_sum: float,
    lambda_start: float,
    delta: float = 0.1,
    eps: float = 1e-7,
    maxiter: int = 1000,
) -> float:
    """
    Finds the lambda for the correction of the physical susceptibility by an iterative scheme (similar to newton).
    """
    lambda_: float = lambda_start + delta

    for _ in range(maxiter):
        chi_lam = apply_lambda(chi_r_mat, lambda_)
        chir_sum = chi_lam.sum(axis=-1).mean() / config.sys.beta
        f_lam = chir_sum - chi_r_loc_sum
        fp_lam = -(chi_lam**2).sum(axis=-1).mean() / config.sys.beta
        lambda_new = lambda_ - (f_lam / fp_lam).real

        if abs(f_lam.real) < eps:
            return lambda_new

        if lambda_new < lambda_:
            delta /= 2
            lambda_ = lambda_start + delta
        else:
            lambda_ = lambda_new

    return lambda_


def perform_single_lambda_correction(chi_r: FourPoint, chi_r_loc: LocalFourPoint) -> tuple[FourPoint, float]:
    """
    Performs the lambda correction on the physical susceptibility for a single spin-channel. Returns (i) the corrected
    susceptibility in the irreducible BZ and half niw range and (ii) lambda as a tuple.
    """
    logger = config.logger
    chi_r = chi_r.to_full_niw_range().map_to_full_bz(config.lattice.q_grid.irrk_inv, config.lattice.q_grid.nk)
    chi_r_mat = chi_r.compress_q_dimension().mat.squeeze()
    lambda_start = get_lambda_start(chi_r_mat)

    lambda_r = find_lambda(chi_r_mat, chi_r_loc.mat.sum() / config.sys.beta, lambda_start)
    chi_r.mat = apply_lambda(chi_r_mat, lambda_r)[config.lattice.q_grid.irrk_ind][:, None, None, None, None, :]
    return chi_r.to_half_niw_range(), lambda_r
