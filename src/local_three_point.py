from local_n_point import LocalNPoint
from n_point_base import *


class LocalThreePoint(LocalNPoint, IHaveChannel):
    def __init__(
        self,
        mat: np.ndarray,
        channel: Channel = Channel.NONE,
        num_bosonic_frequency_dimensions: int = 1,
        num_fermionic_frequency_dimensions: int = 1,
        full_niw_range: bool = True,
        full_niv_range: bool = True,
    ):
        LocalNPoint.__init__(
            self,
            mat,
            4,
            num_bosonic_frequency_dimensions,
            num_fermionic_frequency_dimensions,
            full_niw_range,
            full_niv_range,
        )
        IHaveChannel.__init__(self, channel)
