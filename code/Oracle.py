from abc import ABC, abstractmethod
from typing import Iterable
from PIL import Image


class OracleInterface(ABC):
    """Abstract Oracle class. Your oracle should inherit from this."""

    @abstractmethod
    def ask(
        prompt: str,
        images=Iterable["Image"],
    ):
        # TODO: this is what you have to implement
        pass
