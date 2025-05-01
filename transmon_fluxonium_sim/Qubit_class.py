from matplotlib.pylab import eig
import numpy as np
import scipy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from abc import ABC, abstractmethod
import qutip as qt

e = 1
hbar = 1
kB = 1
h = 1

# hbar = 1.05457e-34  # Reduced Planck's constant (J·s)
# e = 1.602e-19  # Elementary charge (C)
# kB = 1.38e-23  # Boltzmann constant (J/K)
# h = 6.626e-34  # Planck's constant (J·s)

# ---------------------------------------------------------------------------------
# Qubit Parent class
# ----------------------------------------------------------------------------------


class Qubit(ABC):
    def __init__(
        self,
        N: int = None,
        phi_range: float = None,
        basis: str = None,
        init_hamiltonian: bool = True,  # Flag to control when to calculate the Hamiltonian
    ):
        self._N = N if N is not None else 1001
        self._phi_range = phi_range if phi_range is not None else 4 * np.pi
        self._basis = basis  # Basis is set by child classes
        self.eig_vals = None  # Store eigenvalues here
        self.eig_vecs = None  # Store eigenvectors here

        self._gen_operators()  # Generate basis matrices only once or when N changes

        self._update_basis(
            calc_hamiltonian=False
        )  # Update the basis when N or phi_range changes
        if init_hamiltonian:
            self._calc_H()  # Only calculate if flag is True

    def _gen_operators(self):
        """
        Generate the discretization arrays once so they can be reused
        in the flux and charge basis calculations.
        """
        self._diag_base = np.linspace(-1, 1, self._N)
        self._off_diag = np.ones(self._N - 1)
        self._diag = np.ones(self._N)
        # print(f"Generated self._diag_base = {self._diag_base}")

    def _update_basis(self, calc_hamiltonian=True):
        """
        Recompute basis matrices when basis, N, or phi_range changes.
        Also update the discretization operators.
        """
        # If parameters like N or phi_range change, update the operators
        self._gen_operators()
        # print(f"Updating basis: {self._basis}")

        # Use the appropriate basis based on self._basis
        if self._basis == "flux":
            self.Phi, self.n, self.n_2 = self.flux_basis()
        elif self._basis == "charge":
            self.Phi, self.n, self.n_2 = self.charge_basis()
        else:
            raise ValueError(f"Unknown basis: {self._basis}")

        if calc_hamiltonian:
            self._calc_H()  # Only calculate if flag is True

    def _calc_H(self):
        """Compute Hamiltonian and store eigenvalues/eigenvectors."""
        self.eig_vals, self.eig_vecs = self.hamiltonian()

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value: int):
        self._N = value
        self._update_basis()

    @property
    def phi_range(self):
        return self._phi_range

    @phi_range.setter
    def phi_range(self, value: float):
        self._phi_range = value
        self._update_basis()

    @property
    def basis(self):
        return self._basis

    @basis.setter
    def basis(self, value: str):
        if value not in ["flux", "charge"]:
            raise ValueError("Basis must be 'flux' or 'charge'")
        self._basis = value
        self._update_basis()

    def flux_basis(self):
        """Construct flux basis operators. in the aumann paper, they call it Phi, q, q_2, but we call it n, n_2"""
        delta = np.diff(self._phi_range / 2 * self._diag_base)[0]  # Step size
        print(f"Delta: {delta}")
        print(f"delta squared: {delta**2}")
        print(f"1/delta squared: {1/delta**2}")
        self.phi = (self._phi_range / 2) * self._diag_base  # phi is a list.
        print(f"phi: {self.phi}")
        Phi = np.diag(self.phi)  # Phi is a diag matrix
        print(f"Phi: {Phi}")
        # print(f"Phi: {Phi}")
        n = (-1j * hbar / (2 * delta)) * (
            -np.diag(self._off_diag, -1) + np.diag(self._off_diag, 1)
        )
        # print(f"q: {q}")
        # n = q / (2 * e)  # Charge operator in flux basis
        # print(f"n: {n}")
        self.off_diag_base = np.linspace(-1, 1, self._N - 1)
        # print(f"q_2: {q_2}")
        n_2 = (-(hbar**2) / delta**2) * (
            np.diag(self._diag * -2)
            + np.diag(self._off_diag, -1)
            + np.diag(self._off_diag, 1)
        )
        # print(f"q_2: {q_2}")
        # n_2 = q_2 / (4 * e**2)  # Charge operator in flux basis
        # print(f"n_2: {n_2}")
        return Phi, n, n_2

    def charge_basis(self):
        """Construct charge basis operators."""
        self.n_range = np.arange(
            -(self._N // 2), (self._N // 2) + 1
        )  # Charge basis states with integers between -N and N
        # print(f"n_range = {n_range}")
        q = (2 * e) * np.diag(self.n_range)
        # print(f"q: {q}")
        n = q / (2 * e)
        # print(f"n: {n}")
        q_2 = (4 * e**2) * np.diag((self.n_range) ** 2)
        # print(f"q_2: {q_2}")
        n_2 = q_2 / (4 * e**2)
        # print(f"n_2: {n_2}")
        self.phi = (self._phi_range / 2) * self._diag_base  # phi is a list.
        Phi = 0.5 * (
            np.diag(self._off_diag, -1) + np.diag(self._off_diag, 1)
        )  # Obs! This is already cos(phi) in the charge basis
        # print(f"Phi: {Phi}")
        return Phi, n, n_2

    def change_basis_eig_vecs(self, k: int = 0):
        """Change the basis of the qubit eigenvectors."""
        print(f"Changing basis from {self.basis}")

        if self.basis == "charge":
            # Transformation from charge basis to flux basis
            # |φ⟩ = ∑ eⁱⁿᵠ |n⟩
            U = np.exp(
                1j * (self.n_range[:, None] * self.phi[None, :])
            )  # Transformation matrix
            self.U = U
            transformed_eig_vecs = np.dot(U, self.eig_vecs)

            # phi = np.linspace(-np.pi, np.pi, 1001)

            print(f"phi = {self.phi}")
            print(f"self.phi = {self.phi}")
            print(f"self.n_range = {self.n_range}")
            print(f"self.eig_vecs = {self.eig_vecs[k]}")
            psi = []
            for i, val in enumerate(self.n_range):
                print(f"val = {val}")
                print(f"i = {i}")
                psi.append(self.eig_vecs[k][i] * np.exp(1j * val * self.phi))
                print(
                    f"psi component shape = {(self.eig_vecs[k][i] * np.exp(1j * val * self.phi)).shape}"
                )
                print(f"psi = {psi}")
            psi = np.array(psi)
            print(f"psi = {psi}")
            # Sum over Fourier components to get eigenwave
            psi = np.sum(psi, axis=0) / np.sqrt(2 * np.pi)
            print(f"psi: {psi}")
            # Normalize Psi
            norm = np.sqrt(np.dot(psi, psi.conj()))
            print(f"norm: {norm}")
            psi = psi / norm
            print(f"psi: {psi}")
            return psi, self.phi

        elif self.basis == "flux":
            # Transformation from flux basis to charge basis
            # |n⟩ = (1/2π) ∫ e⁻ⁱⁿᵠ |φ⟩ dφ
            # In discrete form, this becomes a matrix with elements (1/2π)e⁻ⁱⁿᵠ
            U = np.exp(-1j * (self.n_range[None, :] * self.phi[:, None])).T / (
                2 * np.pi
            )
            transformed_eig_vecs = np.dot(U, self.eig_vecs)

        else:
            raise ValueError(f"Unknown basis: {self.basis}")

        # Normalize the transformed eigenvectors (common for both cases)
        norms = np.sqrt(np.sum(np.abs(transformed_eig_vecs) ** 2, axis=0))
        normalized_eig_vecs = transformed_eig_vecs / norms[np.newaxis, :]

        return normalized_eig_vecs

    @abstractmethod
    def potential(self):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self):
        raise NotImplementedError

    def anharmonicity(self):
        eig_vals, _ = self.hamiltonian()
        if len(eig_vals) < 3:
            raise ValueError("Not enough eigenvalues to compute anharmonicity")
        return (eig_vals[2] - eig_vals[1]) - (eig_vals[1] - eig_vals[0])

    def n_ij(self, i: int, j: int):
        """Return the matrix element of the charge operator in the energy eigenbasis."""
        return np.dot(self.eig_vecs[i].conj(), np.dot(self.n, self.eig_vecs[j]))

    def Phi_ij(self, i: int, j: int):
        """Return the matrix element of the flux operator in the energy eigenbasis."""
        return np.dot(self.eig_vecs[i].conj(), np.dot(self.Phi, self.eig_vecs[j]))

    def n_qutip(self, n_levels: int, thresh=1e-4):
        """Convert the charge operator to Qutip in the energy eigenbasis."""
        # eig_vecs = self.eig_vecs[
        #     :, :n_levels
        # ]  # Take only the first n_levels eigenvectors
        # n_matrix = eig_vecs.T @ self.n @ eig_vecs  # Transform n into energy eigenbasis
        # n_matrix[np.abs(n_matrix) < thresh] = 0  # Apply thresholding
        n_op = 1j * np.zeros((n_levels, n_levels))
        # print(f"n_op: {n_op}")
        for i in range(n_levels):
            for j in range(n_levels):
                if i == j:
                    n_op[i, j] = 0
                else:
                    val = self.n_ij(i, j)
                    if thresh is not None:
                        if np.abs(val) < thresh:
                            val = 0
                            n_op[i, j] = val
                        n_op[i, j] = val
        # print(f"n_op: {n_op}")
        return qt.Qobj(n_op)

    def Phi_qutip(self, n_levels: int, thresh=1e-4):
        """Convert the flux operator to Qutip in the energy eigenbasis."""
        # eig_vecs = self.eig_vecs[
        #     :, :n_levels
        # ]  # Take only the first n_levels eigenvectors
        # phi_matrix = eig_vecs.T @ self.Phi @ eig_vecs  # Transform Phi into energy eigenbasis
        # phi_matrix[np.abs(phi_matrix) < thresh] = 0  # Apply thresholding
        phi_op = np.zeros((n_levels, n_levels))
        for i in range(n_levels):
            for j in range(n_levels):
                val = self.Phi_ij(i, j)
                if thresh is not None:
                    if np.abs(val) < thresh:
                        val = 0
                        phi_op[i, j] = val
                    phi_op[i, j] = val
        return qt.Qobj(phi_op)

    def hamiltonian_qutip(self, n_levels=10):
        """Return the Hamiltonian as a Qutip operator."""
        H = qt.Qobj(np.diag(self.eig_vals[:n_levels] - self.eig_vals[0]))
        return H


# ------------------------------------------------------------------------------
# Fluxonium Qubit child class
# ------------------------------------------------------------------------------
class Fluxonium(Qubit):
    """This class is a child class of the Qubit class. It is a fluxonium qubit."""

    def __init__(
        self,
        E_C: float = 1.0,
        E_J: float = 1.0,
        E_L: float = 1.0,
        phi_ext: float = np.pi,
        N: int = None,
        basis: str = "flux",  # Default basis for Fluxonium
        phi_range: float = None,
    ):
        # Initialize with hamiltonian calculation disabled
        super().__init__(N, phi_range, basis, init_hamiltonian=False)
        # Set specific attributes
        self._E_C = E_C
        self._E_J = E_J
        self._E_L = E_L
        self._phi_ext = phi_ext
        # Now calculate the Hamiltonian
        self._calc_H()

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value: float):
        self._E_C = value
        self._calc_H()

    @property
    def E_J(self):
        return self._E_J

    @E_J.setter
    def E_J(self, value: float):
        self._E_J = value
        self._calc_H()

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, value: float):
        self._E_L = value
        self._calc_H()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value: float):
        self._phi_ext = value
        self._calc_H()

    @property
    def potential(self):
        return (
            -self.E_J * np.cos(np.diag(self.Phi) - self.phi_ext)
            + 0.5 * self.E_L * np.diag(self.Phi) ** 2
        )

    def hamiltonian(self):
        # capacitance term
        print(f"Phi: {self.Phi}")
        print(f"Phi_ext: {self.phi_ext}")
        print(f"cos(Phi) = {np.cos(np.diag(self.Phi) - self.phi_ext)}")
        C = np.dot(4 * self.E_C, self.n_2)
        print(f"Charge Hamiltonian: {C}")

        # Josephson energy term
        if self._basis == "flux":
            # In flux basis, Phi is diagonal => apply cos(φ - φ_ext) directly
            JJ = np.diag(self.E_J * np.cos(np.diag(self.Phi) - self.phi_ext))
            # Inductive energy: (1/2) * E_L * phi^2
            inductor = np.diag(0.5 * self.E_L * np.diag(self.Phi) ** 2)
        else:
            # In charge basis, Phi is the cos(φ) operator already (hopping matrix)
            JJ = self.E_J * self.Phi
            # No flux operator for inductive term in charge basis
            inductor = np.zeros_like(JJ)
        print(f"Josephson junction Hamiltonian: {- JJ}")
        print(f"Josephson junction Hamiltonian.shape: {JJ.shape}")
        print(f"Inductor Hamiltonian: {inductor}")
        print(f"Inductor Hamiltonian.shape: {inductor.shape}")

        # Total Hamiltonian
        H = C - JJ + inductor
        print(f"Total Hamiltonian: {H}")
        # Generate and correct eigenvalues and eigenvectors
        eig_vals, eig_vecs = LA.eigh(H)
        eig_vals = eig_vals - eig_vals[0]  # Shift eigenvalues to start from zero
        eig_vecs = eig_vecs.T
        for j in range(len(eig_vecs)):
            # Find the index of the maximum absolute value in the eigenvector
            max_idx = np.argmax(np.abs(eig_vecs[j]))
            if not np.isclose(eig_vecs[j][max_idx], 0):
                # Use the phase of the maximum component for correction
                phase = eig_vecs[j][max_idx] / np.abs(eig_vecs[j][max_idx])
                eig_vecs[j] = eig_vecs[j] / phase
        return eig_vals, eig_vecs


# ------------------------------------------------------------------------------
# Transmon Qubit child class
# ------------------------------------------------------------------------------
class Transmon(Qubit):
    """This class is a child class of the Qubit class. It is a flux-tunable transmon qubit."""

    def __init__(
        self,
        E_C: float = 1.0,
        E_J1: float = 1.0,
        E_J2: float = 1.0,
        E_L: float = 0.0,
        ng: float = 0.5,
        N: int = None,
        phi_ext: float = 0.0,
        basis: str = "charge",
        phi_range: float = None,
    ):
        # Initialize with hamiltonian calculation disabled
        super().__init__(N, phi_range, basis, init_hamiltonian=False)
        # Set specific attributes
        self._E_C = E_C
        self._E_J1 = E_J1
        self._E_J2 = E_J2
        self._E_L = E_L
        self._ng = ng
        self._phi_ext = phi_ext
        # Now calculate the Hamiltonian
        self._calc_H()

    @property
    def E_C(self):
        return self._E_C

    @E_C.setter
    def E_C(self, value: float):
        self._E_C = value
        self._calc_H()

    @property
    def E_J1(self):
        return self._E_J1

    @E_J1.setter
    def E_J1(self, value: float):
        self._E_J1 = value
        self._calc_H()

    @property
    def E_J2(self):
        return self._E_J2

    @E_J2.setter
    def E_J2(self, value: float):
        self._E_J2 = value
        self._calc_H()

    @property
    def E_J(self):
        """Compute total Josephson energy dynamically from E_J1 and E_J2."""
        return self.E_J1 + self.E_J2

    @property
    def gamma(self):
        """Compute gamma dynamically from E_J1 and E_J2."""
        return self.E_J2 / self.E_J1

    @property
    def d(self):
        """Compute asymmetry parameter d dynamically."""
        return (self.gamma - 1) / (self.gamma + 1)

    @property
    def ng(self):
        return self._ng

    @ng.setter
    def ng(self, value: float):
        self._ng = value
        self._calc_H()

    @property
    def phi_ext(self):
        return self._phi_ext

    @phi_ext.setter
    def phi_ext(self, value: float):
        self._phi_ext = value
        self._calc_H()

    @property
    def potential(self):
        # if self._basis == "flux":
        # Potential energy term: E_J * cos(phi - phi_ext)
        return -self.E_J * np.cos(self.phi)
        # else:
        #     return -np.dot(self.E_J, np.diag(self.Phi))

    def hamiltonian(self):
        # Josephson energy term: E_J * cos(phi - phi_ext)
        if self._basis == "flux":
            # Charging energy term: 4E_C(n - n_g)^2
            # C = np.diag(np.dot(4 * self.E_C, (np.diag(self.n) - self.ng) ** 2))
            C = np.dot(4 * self.E_C, (self.n_2 - 2 * np.dot(self.n, self.ng)))
            print(f"Capacitance part: {C}")
            # C = np.dot(4 * self.E_C, self.n_2)
            # print(f"Capacitance part: {C}")
            # Phi is diagonal in flux basis -> use cos(Phi)
            JJ = np.diag(-self.E_J * np.cos(self.phi - self.phi_ext))
            print(f"Josephson part: {JJ}")
            C = (
                4
                * self.E_C
                * (self.n_2 - 2 * self.ng * self.n + self.ng**2 * np.eye(self.N))
            )
            print(f"Capacitance part: {C}")
            JJ = -self.E_J * np.diag(np.cos(np.diag(self.Phi) - self.phi_ext))
            print(f"Josephson part: {JJ}")
        else:
            # Charging energy term: 4E_C(n - n_g)^2
            C = np.diag(np.dot(4 * self.E_C, (np.diag(self.n) - self.ng) ** 2))
            print(f"Capacitance part: {C}")
            # Phi is cos(φ) operator in charge basis
            JJ = -self.E_J * self.Phi  # Here Phi is the cos(φ) operator
        # print(f"Capacitance part: {C}")
        # print(f"Josephson part: {JJ}")

        # Total hamiltonian
        H = C + JJ
        print(f"Total Hamiltonian: {H}")

        # Calculate and correct eigenvalues and eigenvectors
        eig_vals, eig_vecs = LA.eigh(H)
        eig_vals = eig_vals - eig_vals[0]  # Shift eigenvalues to start from zero
        eig_vecs = eig_vecs.T
        # for j in range(len(eig_vecs)):
        #     # Find the index of the maximum absolute value in the eigenvector
        #     max_idx = np.argmax(np.abs(eig_vecs[j]))
        #     if not np.isclose(eig_vecs[j][max_idx], 0):
        #         # Use the phase of the maximum component for correction
        #         phase = eig_vecs[j][max_idx] / np.abs(eig_vecs[j][max_idx])
        #         eig_vecs[j] = eig_vecs[j] / phase
        return eig_vals, eig_vecs


# ------------------------------------------------------------------------------
# Define a helper class for two coupled qubits - not using
class CoupledQubits:
    def __init__(self, qubit1, qubit2, g, n_levels=5):
        """
        qubit1, qubit2: instances of Transmon or Fluxonium.
        g: coupling strength (assumed to be given in the same frequency units).
        n_levels: number of levels to use in the truncated energy basis.
        """
        self.n_levels = n_levels
        self.g = g
        # Get the single-qubit Hamiltonians as Qutip objects.
        # These are diagonal (energy eigenbasis, with ground state shifted to zero).
        self.H1 = qubit1.hamiltonian_qutip(n_levels)
        self.H2 = qubit2.hamiltonian_qutip(n_levels)
        # Use the standard annihilation operators in a Fock space of dimension n_levels.
        self.a1 = qt.destroy(n_levels)
        self.a2 = qt.destroy(n_levels)

        # Build the composite Hamiltonian (without any additional offset)
        self.H_compound = qt.tensor(self.H1, qt.qeye(n_levels)) + qt.tensor(
            qt.qeye(n_levels), self.H2
        )
        # Coupling term: g*(a1^dagger ⊗ a2 + a1 ⊗ a2^dagger)
        self.H_coup = self.g * (
            qt.tensor(self.a1.dag(), self.a2) + qt.tensor(self.a1, self.a2.dag())
        )
        self.H_total = self.H_compound + self.H_coup

    def add_offset(self, delta):
        """
        Add an offset delta to the first qubit’s Hamiltonian.
        (This allows us to sweep the detuning Δ = ωₐ - ω_b.)
        """
        H1_offset = self.H1 + delta * qt.qeye(self.n_levels)
        H_compound = qt.tensor(H1_offset, qt.qeye(self.n_levels)) + qt.tensor(
            qt.qeye(self.n_levels), self.H2
        )
        self.H_total = H_compound + self.H_coup

    def eigenenergies(self):
        # Returns the sorted eigenenergies of the full Hamiltonian.
        return np.sort(self.H_total.eigenenergies())
