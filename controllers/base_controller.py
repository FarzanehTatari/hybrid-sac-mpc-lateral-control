from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def control(self, x: np.ndarray, k: int, **kwargs) -> float:
        pass
