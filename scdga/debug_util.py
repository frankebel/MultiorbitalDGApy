import itertools

import numpy as np

from scdga import config
from scdga.four_point import FourPoint
from scdga.greens_function import GreensFunction
from scdga.interaction import LocalInteraction
from scdga.local_four_point import LocalFourPoint
from scdga.n_point_base import IHaveMat
from scdga.self_energy import SelfEnergy


def _count_nonzero_orbital_entries(obj, obj_name: str, num_orbital_dimensions: int) -> int:
    count = 0
    indices = itertools.product(range(config.sys.n_bands), repeat=num_orbital_dimensions)
    for idx in indices:
        val = (
            (obj.mat[:, *idx] if obj.has_compressed_q_dimension else obj.mat[:, :, :, *idx])
            if (isinstance(obj, (FourPoint, SelfEnergy, GreensFunction)))
            else obj.mat[idx]
        )

        if not np.allclose(np.abs(val.max()), 0.0 + 0.0j) and not np.allclose(np.abs(val.min()), 0.0 + 0.0j):
            config.logger.log_debug(
                f"{idx} is non-zero for {obj_name} with max = {val.max():.6f} and min = {val.min():.6f}"
            )
            count += 1
    config.logger.log_debug(f"Number of non-zero elements in {obj_name}: {count}")
    return count


def count_nonzero_orbital_entries(obj: IHaveMat, obj_name: str) -> int:
    if config.sys.n_bands <= 1:
        return -1
    if isinstance(obj, (FourPoint, LocalFourPoint, LocalInteraction)):
        return _count_nonzero_orbital_entries(obj, obj_name, num_orbital_dimensions=4)
    elif isinstance(obj, SelfEnergy) or isinstance(obj, GreensFunction):
        return _count_nonzero_orbital_entries(obj, obj_name, num_orbital_dimensions=2)
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")
