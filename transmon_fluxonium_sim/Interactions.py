import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import scipy.linalg as LA


# Function to create the total Hamiltonian for two coupled systems -----------
def total_hamiltonian_cap(h1, h2, J, n):
    """
    Create the total Hamiltonian for two coupled qubits which are capacitively coupled.
    Parameters:
    H1 (qutip.Qobj): Hamiltonian of the first system.
    H2 (qutip.Qobj): Hamiltonian of the second system.
    J (float): Coupling strength.
    n_levels (int): Number of states in the Hilbert space.

    Returns:
    qutip.Qobj: Total Hamiltonian.
    """
    H1 = h1.hamiltonian_qutip(n_levels=n)
    # print(f"H1: {H1}")
    H2 = h2.hamiltonian_qutip(n_levels=n)
    # print(f"H2: {H2}")
    n_H1 = h1.n_qutip(n_levels=n)
    # print(f"q_H1: {n_H1}")
    n_H2 = h2.n_qutip(n_levels=n)
    # print(f"q_H2: {n_H2}")
    h_id = qt.qeye(n)
    # print(f"h_id: {h_id}")
    H1_qt = qt.tensor([H1, h_id])
    # print(f"H1_qt: {H1_qt}")
    H2_qt = qt.tensor([h_id, H2])
    # print(f"H2_qt: {H2_qt}")
    H_coupling = J * qt.tensor([n_H1, n_H2])
    # print(f"H_coupling: {H_coupling}")
    H_total = 2 * np.pi * (H1_qt + H2_qt + H_coupling)
    # print(f"H_total: {H_total}")
    return H_total


def FloquetCoeff(t, args):
    # We solved the static Hamiltonian at t=0 to find an energy eigenbasis
    # We now want to subtract off the static component, and explicitly add it back in
    # as a time component    [+H(t=0) - H(t=0)] + H(t)

    EL = args["EL"]
    omega = args["omega"]
    alpha = args["alpha"]

    coeff = -EL * alpha * np.cos(omega * t)
    return coeff


def FloquetH(qubit1, q_states=4):
    h_id = qt.qeye(q_states)
    H_floq = 2 * np.pi * qt.tensor([h_id, qubit1.phi_to_qutip(q_states)])
    return H_floq


# Function to compute ZZ coupling from the lowest four eigenenergies. ----------
def compute_zz(energies):
    # energies is an array (sorted in ascending order)
    E00, E10, E01, E11 = energies[0], energies[1], energies[2], energies[3]
    return (E11 - E01) - (E10 - E00)  # in GHz


def get_max_overlap_indices(evecs, targs, prev_evecs=None):
    overlap_idx = []
    if prev_evecs is None:
        # First iteration: Use maximum overlap with target states
        for targ_state in targs:
            overlaps = [np.abs(targ_state.overlap(vec)) ** 2 for vec in evecs]
            max_idx = np.argmax(overlaps)
            overlap_idx.append(max_idx)
    else:
        # Subsequent iterations: Use maximum overlap with previous eigenstates
        for prev_vec in prev_evecs:
            overlaps = [np.abs(prev_vec.overlap(vec)) ** 2 for vec in evecs]
            max_idx = np.argmax(overlaps)
            overlap_idx.append(max_idx)
    return overlap_idx


def gate_freqs(eigs):
    CZ20 = np.abs(eigs[3] - eigs[5])
    CZ02 = np.abs(eigs[3] - eigs[4])
    iswap = np.abs(eigs[2] - eigs[1])
    bswap = np.abs(eigs[3] - eigs[0])
    return np.array([CZ20, CZ02, iswap, bswap])


# # Helper function to find the most probable states. -------------------------------
# def get_max_overlap_indices(evecs, targs):
#     overlap_idx = []
#     for targ_state in targs:
#         for i, vec in enumerate(evecs):
#             if np.abs(targ_state.overlap(vec)) ** 2 >= 0.5:
#                 overlap_idx.append(i)
#     return overlap_idx


# def plot_zz_vs_E_J(
#     transmon, fluxonium, J, n_levels, E_J_range, save_plot=False, filename="zz_vs_E_J"
# ):
#     """
#     Plot ZZ coupling as a function of transmon E_J parameter.

#     Parameters:
#     transmon (Transmon): Transmon qubit instance
#     fluxonium (Fluxonium): Fluxonium qubit instance
#     J (float): Coupling strength
#     n_levels (int): Number of energy levels to consider
#     E_J_range (array): Range of E_J values to sweep
#     save_plot (bool): Whether to save the plot
#     filename (str): Filename to save the plot

#     Returns:
#     tuple: (E_J_values, zz_values)
#     """
#     zz_values = []

#     # Store original E_J value to restore later
#     original_E_J1 = transmon.E_J1

#     for E_J in E_J_range:
#         # Update transmon E_J
#         transmon.E_J1 = E_J

#         # Calculate total Hamiltonian
#         H_total = total_hamiltonian_cap(transmon, fluxonium, J, n_levels)

#         # Get eigenenergies
#         eig_vals, _ = H_total.eigenstates(eigvals=4)  # Get lowest 4 eigenvalues

#         # Calculate ZZ coupling
#         zz = compute_zz(eig_vals)
#         zz_values.append(zz)

#     # Restore original E_J
#     transmon.E_J1 = original_E_J1

#     # Plot results
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(E_J_range, zz_values, linewidth=2)
#     ax.set_xlabel(r"Transmon $E_J$ (GHz)", fontsize=14)
#     ax.set_ylabel(r"ZZ coupling (MHz)", fontsize=14)
#     ax.set_title(r"ZZ coupling vs Transmon $E_J$", fontsize=16)
#     ax.grid(True)

#     # Convert to MHz for better readability
#     zz_values_MHz = np.array(zz_values) / (2 * np.pi * 1e-6)
#     ax.set_ylabel(r"ZZ coupling (MHz)", fontsize=14)

#     if save_plot:
#         plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")

#     plt.show()

#     return E_J_range, zz_values


# def plot_eig_valsnergies_vs_E_J(
#     transmon,
#     fluxonium,
#     J,
#     n_levels,
#     E_J_range,
#     save_plot=False,
#     filename="eigenenergies_vs_E_J",
# ):
#     """
#     Plot eigenenergies of the total Hamiltonian as a function of transmon E_J parameter.

#     Parameters:
#     transmon (Transmon): Transmon qubit instance
#     fluxonium (Fluxonium): Fluxonium qubit instance
#     J (float): Coupling strength
#     n_levels (int): Number of energy levels to consider
#     E_J_range (array): Range of E_J values to sweep
#     save_plot (bool): Whether to save the plot
#     filename (str): Filename to save the plot

#     Returns:
#     tuple: (E_J_values, eigenenergies)
#     """
#     eigenenergies = []

#     # Store original E_J value to restore later
#     original_E_J1 = transmon.E_J1

#     for E_J in E_J_range:
#         # Update transmon E_J
#         transmon.E_J1 = E_J

#         # Calculate total Hamiltonian
#         H_total = total_hamiltonian_cap(transmon, fluxonium, J, n_levels)

#         # Get eigenenergies
#         eigenE, _ = H_total.eigenstates(eigvals=n_levels)  # Get eigenvalues
#         eigenenergies.append(eigenE)

#     # Restore original E_J
#     transmon.E_J1 = original_E_J1

#     # Convert eigenenergies to a numpy array for easier plotting
#     eigenenergies = np.array(eigenenergies)

#     # Plot results
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for i in range(eigenenergies.shape[1]):  # Loop over each eigenenergy
#         ax.plot(E_J_range, eigenenergies[:, i], label=f"Level {i}", linewidth=2)

#     ax.set_xlabel(r"Transmon $E_J$ (GHz)", fontsize=14)
#     ax.set_ylabel(r"Eigenenergies (GHz)", fontsize=14)
#     ax.set_title(r"Eigenenergies vs Transmon $E_J$", fontsize=16)
#     ax.legend(fontsize=12)
#     ax.grid(True)

#     if save_plot:
#         plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
