from enum import Enum
from abc import ABC


class Channel(Enum):
    DENS: str = "dens"
    MAGN: str = "magn"
    NONE: str = "none"


class IHaveChannel(ABC):
    """
    Interface for classes that have a channel attribute.
    """

    def __init__(self, channel: Channel = Channel.NONE):
        self._channel = channel

    @property
    def channel(self) -> Channel:
        return self._channel
