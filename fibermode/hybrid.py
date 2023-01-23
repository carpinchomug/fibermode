from typing import Optional

import numpy as np
import scipy.optimize as optimize
from numpy.typing import ArrayLike
from scipy.constants import epsilon_0, mu_0
from scipy.special import jv, jvp, kv, kvp

from .direction import Direction
from .field import Field
from .mode_base import ModeBase

epsilon0 = epsilon_0
mu0 = mu_0


class HybridModeBase(ModeBase):
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
        super().__init__(
            wavelength,
            radius,
            core_index,
            clad_index,
            azimuthal_order,
            radial_order,
            power,
            propagation_direction,
        )
        self._A = None
        self._power_inside = None
        self._power_outside = None

    @property
    def h(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return np.sqrt((self.n1 * self.k) ** 2 - self.beta**2)

    @property
    def q(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return np.sqrt(self.beta**2 - (self.n2 * self.k) ** 2)

    @property
    def s(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            ha = self.h * self.a
            qa = self.q * self.a
            return (
                self.l
                * (1 / ha**2 + 1 / qa**2)
                / (
                    jvp(self.l, ha) / (ha * jv(self.l, ha))
                    + kvp(self.l, qa) / (qa * kv(self.l, qa))
                )
            )

    @property
    def s1(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return self.beta**2 / (self.k * self.n1) ** 2 * self.s

    @property
    def s2(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return self.beta**2 / (self.k * self.n2) ** 2 * self.s

    @property
    def A(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return self.P / (self._P_in(1) + self._P_out(1))

    @property
    def P_in(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return self._P_in(self.A)

    @property
    def P_out(self) -> Optional[float]:
        if self.beta is None:
            return None
        else:
            return self._P_out(self.A)

    def _eve_(self, ha, qa):
        return (
            jv(self.l - 1, ha) / (ha * jv(self.l, ha))
            + ((self.n1**2 + self.n2**2) / (2 * self.n1**2))
            * (kvp(self.l, qa) / (qa * kv(self.l, qa)))
            - self.l / ha**2
        )

    def _R2(self, ha, qa):
        beta2 = (self.n1 * self.k) ** 2 - (ha / self.a) ** 2
        return ((self.n1**2 - self.n2**2) / (2 * self.n1**2)) ** 2 * (
            kvp(self.l, qa) / (qa * kv(self.l, qa))
        ) ** 2 + beta2 * (self.l / (self.n1 * self.k)) ** 2 * (
            1 / qa**2 + 1 / ha**2
        )

    def _P_in(self, A):
        ha = self.h * self.a
        return (
            self.f
            * A
            * (
                np.pi
                * self.a**2
                * self.omega
                * epsilon0
                * self.n1**2
                * self.beta
                / (4 * self.h**2)
            )
            * (
                (1 - self.s)
                * (1 - self.s1)
                * (jv(self.l - 1, ha) ** 2 - jv(self.l - 2, ha) * jv(self.l, ha))
                + (1 + self.s)
                * (1 + self.s1)
                * (jv(self.l + 1, ha) ** 2 - jv(self.l + 2, ha) * jv(self.l, ha))
            )
        )

    def _P_in(self, A):
        ha = self.h * self.a
        return (
            self.f
            * A
            * (
                np.pi
                * self.a**2
                * self.omega
                * epsilon0
                * self.n1**2
                * self.beta
                / (4 * self.h**2)
            )
            * (
                (1 - self.s)
                * (1 - self.s1)
                * (jv(self.l - 1, ha) ** 2 - jv(self.l - 2, ha) * jv(self.l, ha))
                + (1 + self.s)
                * (1 + self.s1)
                * (jv(self.l + 1, ha) ** 2 - jv(self.l + 2, ha) * jv(self.l, ha))
            )
        )

    def _P_out(self, A):
        qa = self.q * self.a
        return (
            self.f
            * A
            * (
                np.pi
                * self.a**2
                * self.omega
                * epsilon0
                * self.n2**2
                * self.beta
                / (4 * self.q**2)
            )
            * (jv(self.l, self.h * self.a) ** 2 / kv(self.l, qa) ** 2)
            * (
                (1 - self.s)
                * (1 - self.s2)
                * (kv(self.l - 2, qa) * kv(self.l, qa) - kv(self.l - 1, qa) ** 2)
                + (1 + self.s)
                * (1 + self.s2)
                * (kv(self.l + 2, qa) * kv(self.l, qa) - kv(self.l + 1, qa) ** 2)
            )
        )

    def _ha_to_beta(self, ha):
        return np.sqrt((self.k * self.n1) ** 2 - (ha / self.a) ** 2)

    def _e_rho_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        hr = self.h * rho
        return (
            1j
            * self.A
            * (self.beta / (2 * self.h))
            * ((1 - self.s) * jv(self.l - 1, hr) - (1 + self.s) * jv(self.l + 1, hr))
        )

    def _e_phi_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        hr = self.h * rho
        return 0j - self.A * (self.beta / (2 * self.h)) * (
            (1 - self.s) * jv(self.l - 1, hr) + (1 + self.s) * jv(self.l + 1, hr)
        )

    def _e_z_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        return 0j + self.A * jv(self.l, self.h * rho)

    def _e_rho_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        qr = self.q * rho
        return (
            1j
            * self.A
            * (self.beta / (2 * self.q))
            * (jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a))
            * ((1 - self.s) * kv(self.l - 1, qr) + (1 + self.s) * kv(1 + self.l, qr))
        )

    def _e_phi_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        qr = self.q * rho
        return 0j - self.A * (self.beta / (2 * self.q)) * (
            jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a)
        ) * ((1 - self.s) * kv(self.l - 1, qr) - (1 + self.s) * kv(1 + self.l, qr))

    def _e_z_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        return 0j + self.A * (
            jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a)
        ) * kv(self.l, self.q * rho)

    def e_rho(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            return (
                np.sqrt(2)
                * np.where(
                    rho < self.a,
                    self._e_rho_in(rho, phi),
                    self._e_rho_out(rho, phi),
                )
                * np.cos(self.l * phi - pol)
            )

    def e_phi(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            return (
                1j
                * np.sqrt(2)
                * np.where(
                    rho < self.a,
                    self._e_phi_in(rho, phi),
                    self._e_phi_out(rho, phi),
                )
                * np.sin(self.l * phi - pol)
            )

    def e_z(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            return (
                np.sqrt(2)
                * self.f
                * np.where(
                    rho < self.a,
                    self._e_z_in(rho, phi),
                    self._e_z_out(rho, phi),
                )
                * np.cos(self.l * phi - pol)
            )

    def e_x(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        return self.e_rho(rho, phi, pol) * np.cos(phi) - self.e_phi(
            rho, phi, pol
        ) * np.sin(phi)

    def e_y(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        return self.e_rho(rho, phi, pol) * np.sin(phi) + self.e_phi(
            rho, phi, pol
        ) * np.cos(phi)

    def _h_rho_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        hr = self.h * rho
        return 0j + self.A * (self.omega * epsilon0 * self.n1**2 / (2 * self.h)) * (
            (1 - self.s1) * jv(self.l - 1, hr) + (1 + self.s1) * jv(self.l + 1, hr)
        )

    def _h_phi_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        hr = self.h * rho
        return (
            1j
            * self.A
            * (self.omega * epsilon0 * self.n1**2 / (2 * self.h))
            * ((1 - self.s1) * jv(self.l - 1, hr) - (1 + self.s1) * jv(self.l + 1, hr))
        )

    def _h_z_in(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        return (
            1j
            * self.A
            * (self.beta * self.s / (self.omega * mu0))
            * (jv(self.l, self.h * rho))
        )

    def _h_rho_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        qr = self.q * rho
        return 0j + self.A * (self.omega * epsilon0 * self.n2**2 / (2 * self.q)) * (
            jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a)
        ) * ((1 - self.s2) * kv(self.l - 1, qr) - (1 + self.s2) * kv(self.l + 1, qr))

    def _h_phi_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        qr = self.q * rho
        return (
            1j
            * self.A
            * (self.omega * epsilon0 * self.n2**2 / (2 * self.q))
            * (jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a))
            * ((1 - self.s2) * kv(self.l - 1, qr) + (1 + self.s2) * kv(self.l + 1, qr))
        )

    def _h_z_out(self, rho: ArrayLike, phi: ArrayLike) -> ArrayLike:
        return (
            1j
            * self.A
            * (self.beta * self.s / (self.omega * mu0))
            * (jv(self.l, self.h * self.a) / kv(self.l, self.q * self.a))
            * (jv(self.l, self.q * rho))
        )

    def h_rho(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            1j * np.sqrt(2) * self.f * np.where(
                rho < self.a,
                self._h_rho_in(rho, phi),
                self._h_rho_out(rho, phi),
            ) * np.sin(self.l * phi - pol)

    def h_phi(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            np.sqrt(2) * self.f * np.where(
                rho < self.a,
                self._h_phi_in(rho, phi),
                self._h_phi_out(rho, phi),
            ) * np.cos(self.l * phi - pol)

    def h_z(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        if self.beta is None:
            return self._zeros(rho, phi, pol)
        else:
            1j * np.sqrt(2) * np.where(
                rho < self.a, self._h_z_in(rho, phi), self._h_z_out(rho, phi)
            ) * np.sin(self.l * phi - pol)

    def h_x(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        return self.h_rho(rho, phi, pol) * np.cos(phi) - self.h_phi(
            rho, phi, pol
        ) * np.sin(phi)

    def h_y(self, rho: ArrayLike, phi: ArrayLike, pol: ArrayLike = 0.0) -> Field:
        return self.h_rho(rho, phi, pol) * np.sin(phi) + self.h_phi(
            rho, phi, pol
        ) * np.cos(phi)

    # alias profile functions
    Er, Ep, Ez, Ex, Ey = e_rho, e_phi, e_z, e_x, e_y
    Hr, Hp, Hz, Hx, Hy = h_rho, h_phi, h_z, h_x, h_y


class HE(HybridModeBase):
    def __init__(
        self,
        wavelength: float,
        radius: float,
        core_index: float,
        clad_index: float,
        azimuthal_order: int = 1,
        radial_order: int = 1,
        *,
        power: float = 10e-6,  # 10 mW
        propagation_direction: Direction = Direction.FORWARD,
    ):
        super().__init__(
            wavelength,
            radius,
            core_index,
            clad_index,
            azimuthal_order,
            radial_order,
            power,
            propagation_direction,
        )
        self._propagation_constant = self._solve_eve_for_beta()

    def _eve(self, ha):
        qa = np.sqrt(self.V**2 - ha**2)
        return self._eve_(ha, qa) + np.sqrt(self._R2(ha, qa))

    def _solve_eve_for_beta(self):
        ha = np.linspace(0, self.V, 1000)[1:-1]
        eve = self._eve(ha)

        i_zerocrossings = np.where(eve[:-1] * eve[1:] <= 0)[0]
        # remove zerorcrossings because of singularities
        i_zerocrossings = i_zerocrossings[::2]

        if len(i_zerocrossings) < self.m:
            return None

        i = i_zerocrossings[self.m - 1]

        if eve[i] == 0:
            return self._ha_to_beta(ha[i])
        if eve[i + 1] == 0:
            return self._ha_to_beta(ha[i + 1])

        ha_root = optimize.root_scalar(self._eve, bracket=[ha[i], ha[i + 1]]).root
        return self._ha_to_beta(ha_root)
