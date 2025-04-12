import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.linalg as LA


def total_hamiltonian_cap(h1, h2, J, n_levels):
    """
    Create the total Hamiltonian for two coupled systems.

    Parameters:
    H1 (qutip.Qobj): Hamiltonian of the first system.
    H2 (qutip.Qobj): Hamiltonian of the second system.
    J (float): Coupling strength.
    n_levels (int): Number of states in the Hilbert space.

    Returns:
    qutip.Qobj: Total Hamiltonian.
    """

    H1 = h1.hamiltonian_qutip(n_levels=n_levels)
    print(f"H1: {H1}")
    H2 = h2.hamiltonian_qutip(n_levels=n_levels)
    print(f"H2: {H2}")
    q_H1 = h1.n_qutip(n_levels=n_levels)
    print(f"q_H1: {q_H1}")
    q_H2 = h2.n_qutip(n_levels=n_levels)
    print(f"q_H2: {q_H2}")
    h_id = qt.qeye(n_levels)
    print(f"h_id: {h_id}")
    H1_qt = qt.tensor([H1, h_id])
    print(f"H1_qt: {H1_qt}")
    H2_qt = qt.tensor([h_id, H2])
    print(f"H2_qt: {H2_qt}")

    H_coupling = J * qt.tensor([q_H1, q_H2])
    print(f"H_coupling: {H_coupling}")
    return 2 * np.pi * (H1_qt + H2_qt + H_coupling)


def residual_zz(total_Hamiltonian):
    """Calculate the residual ZZ coupling from the total Hamiltonian."""
    # Get the eigenvalues and eigenvectors of the total Hamiltonian
    eig_vals, eig_vecs = LA.eigh(total_Hamiltonian)
    eig_vecs = eig_vecs.T

    # Calculate the residual ZZ coupling
    residual_zz = np.zeros((len(eig_vals), len(eig_vals)), dtype=complex)
    for i in range(len(eig_vals)):
        for j in range(len(eig_vals)):
            if i != j:
                residual_zz[i, j] = np.dot(eig_vecs[i], eig_vecs[j])

    return residual_zz


# Function to compute ZZ coupling from the lowest four eigenenergies.
def compute_zz(energies):
    # energies is an array (sorted in ascending order)
    E00, E01, E10, E11 = energies[0], energies[1], energies[2], energies[3]
    return (E11 - E01) - (E10 - E00)
