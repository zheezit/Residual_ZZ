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


def gate_freqs(eigs):
    CZ20 = np.abs(eigs[3] - eigs[5])
    CZ02 = np.abs(eigs[3] - eigs[4])
    iswap = np.abs(eigs[2] - eigs[1])
    return np.array([CZ20, CZ02, iswap])


def plot_zz_vs_E_J(
    transmon, fluxonium, J, n_levels, E_J_range, save_plot=False, filename="zz_vs_E_J"
):
    """
    Plot ZZ coupling as a function of transmon E_J parameter.

    Parameters:
    transmon (Transmon): Transmon qubit instance
    fluxonium (Fluxonium): Fluxonium qubit instance
    J (float): Coupling strength
    n_levels (int): Number of energy levels to consider
    E_J_range (array): Range of E_J values to sweep
    save_plot (bool): Whether to save the plot
    filename (str): Filename to save the plot

    Returns:
    tuple: (E_J_values, zz_values)
    """
    zz_values = []

    # Store original E_J value to restore later
    original_E_J1 = transmon.E_J1

    for E_J in E_J_range:
        # Update transmon E_J
        transmon.E_J1 = E_J

        # Calculate total Hamiltonian
        H_total = total_hamiltonian_cap(transmon, fluxonium, J, n_levels)

        # Get eigenenergies
        eig_vals, _ = H_total.eigenstates(eigvals=4)  # Get lowest 4 eigenvalues

        # Calculate ZZ coupling
        zz = compute_zz(eig_vals)
        zz_values.append(zz)

    # Restore original E_J
    transmon.E_J1 = original_E_J1

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_J_range, zz_values, linewidth=2)
    ax.set_xlabel(r"Transmon $E_J$ (GHz)", fontsize=14)
    ax.set_ylabel(r"ZZ coupling (MHz)", fontsize=14)
    ax.set_title(r"ZZ coupling vs Transmon $E_J$", fontsize=16)
    ax.grid(True)

    # Convert to MHz for better readability
    zz_values_MHz = np.array(zz_values) / (2 * np.pi * 1e-6)
    ax.set_ylabel(r"ZZ coupling (MHz)", fontsize=14)

    if save_plot:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")

    plt.show()

    return E_J_range, zz_values


def plot_eig_valsnergies_vs_E_J(
    transmon,
    fluxonium,
    J,
    n_levels,
    E_J_range,
    save_plot=False,
    filename="eigenenergies_vs_E_J",
):
    """
    Plot eigenenergies of the total Hamiltonian as a function of transmon E_J parameter.

    Parameters:
    transmon (Transmon): Transmon qubit instance
    fluxonium (Fluxonium): Fluxonium qubit instance
    J (float): Coupling strength
    n_levels (int): Number of energy levels to consider
    E_J_range (array): Range of E_J values to sweep
    save_plot (bool): Whether to save the plot
    filename (str): Filename to save the plot

    Returns:
    tuple: (E_J_values, eigenenergies)
    """
    eigenenergies = []

    # Store original E_J value to restore later
    original_E_J1 = transmon.E_J1

    for E_J in E_J_range:
        # Update transmon E_J
        transmon.E_J1 = E_J

        # Calculate total Hamiltonian
        H_total = total_hamiltonian_cap(transmon, fluxonium, J, n_levels)

        # Get eigenenergies
        eigenE, _ = H_total.eigenstates(eigvals=n_levels)  # Get eigenvalues
        eigenenergies.append(eigenE)

    # Restore original E_J
    transmon.E_J1 = original_E_J1

    # Convert eigenenergies to a numpy array for easier plotting
    eigenenergies = np.array(eigenenergies)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(eigenenergies.shape[1]):  # Loop over each eigenenergy
        ax.plot(E_J_range, eigenenergies[:, i], label=f"Level {i}", linewidth=2)

    ax.set_xlabel(r"Transmon $E_J$ (GHz)", fontsize=14)
    ax.set_ylabel(r"Eigenenergies (GHz)", fontsize=14)
    ax.set_title(r"Eigenenergies vs Transmon $E_J$", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)

    if save_plot:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
