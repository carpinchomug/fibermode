from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.constants import c

from fibermode.direction import Direction
from fibermode.field import Field


class ModeBase:
    def __init__(
        self,
        wavelength: float,
        radius: float,
        core_index: float,
        clad_index: float,
        azimuthal_order: int,
        radial_order: int,
        power: float,
        propagation_direction: Direction,
    ):
        self._wavelength = wavelength
        self._radius = radius
        self._core_index = core_index
        self._clad_index = clad_index
        self._azimuthal_order = azimuthal_order
        self._radial_order = radial_order
        self._power = power
        self._propagation_direction = propagation_direction
        self._propagation_constant = None

    @property
    def lam(self) -> float:
        return self._wavelength

    @property
    def a(self) -> float:
        return self._radius

    @property
    def n1(self) -> float:
        return self._core_index

    @property
    def n2(self) -> float:
        return self._clad_index

    @property
    def l(self) -> int:
        return self._azimuthal_order

    @property
    def m(self) -> int:
        return self._radial_order

    @property
    def f(self) -> int:
        return self._propagation_direction.value

    @property
    def omega(self) -> float:
        return 2 * np.pi * c / self.lam

    @property
    def k(self) -> float:
        return 2 * np.pi / self.lam * self.f

    @property
    def V(self) -> float:
        return self.k * self.a * np.sqrt(self.n1**2 - self.n2**2)

    @property
    def P(self) -> float:
        return self._power

    @property
    def beta(self) -> Optional[float]:
        return self._propagation_constant * self.f

    @staticmethod
    def _zeros(*args: ArrayLike) -> Field:
        shape = np.broadcast_shapes([np.shape(x) for x in args])
        return np.zeros(shape, dtype=complex)
