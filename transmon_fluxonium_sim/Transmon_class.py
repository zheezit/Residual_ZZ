import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Set periodicity and number of energy levels to plot
PERIODICITY = 2 * np.pi
NUMBER_LEVELS = 5


class TransmonQubit:
    """Class representing a Transmon Qubit system."""

    def __init__(self, E_J=9.0, E_L=1.0, E_C=2.0, phi_ext=np.pi, N=101):
        self.E_J = E_J
        self.E_L = E_L
        self.E_C = E_C
        self.phi_ext = phi_ext
        self.N = N
        self.phi = np.linspace(-PERIODICITY, PERIODICITY, N)

    def potential(self):
        """Compute the transmon potential with coordinate transformation."""
        return (
            -self.E_J * np.cos(self.phi - self.phi_ext) + 0.5 * self.E_L * self.phi**2
        ) + 2.7

    def hamiltonian(self):
        """Compute the transmon Hamiltonian and return eigenvalues & eigenvectors."""
        delta = self.phi[1] - self.phi[0]  # Step size
        e = 1  # Electron charge in natural units

        # Charge term (kinetic energy)
        q_2 = (
            np.diag(-2 * np.ones(self.N))
            + np.diag(np.ones(self.N - 1), -1)
            + np.diag(np.ones(self.N - 1), 1)
        )
        q_2 *= -1 / delta**2
        C = 4 * self.E_C * q_2

        # Josephson energy term
        JJ = np.diag(self.E_J * np.cos(self.phi - self.phi_ext))

        # Inductive energy term
        inductor = np.diag(0.5 * self.E_L * self.phi**2)

        # Hamiltonian = kinetic - Josephson + inductive
        H = C - JJ + inductor

        # Solve eigenvalues and eigenvectors
        eig_vals, eig_vecs = LA.eigh(H)
        return eig_vals, eig_vecs


def plot_transmon_fixed(transmon):
    """Plot transmon potential and eigenenergies for fixed values."""
    fig, ax = plt.subplots()

    # Compute potential and eigenvalues
    potential = transmon.potential()
    eig_vals, eig_vecs = transmon.hamiltonian()

    # Plot potential
    ax.plot(transmon.phi, potential, label="Potential", color="black")

    # Plot wavefunctions (scaled) and eigenenergies
    for i in range(NUMBER_LEVELS):
        ax.plot(transmon.phi, 5 * eig_vecs[:, i] + eig_vals[i], label=f"ψ_{i}")
        ax.axhline(y=eig_vals[i], linestyle="--", lw=1, color="gray")

    # Formatting
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"Energy/$h$ (GHz)")
    ax.set_ylim([min(potential) - 1, max(eig_vals) + 2])
    ax.legend()
    plt.show()


def plot_transmon_interactive(transmon):
    """Interactive plot with sliders to adjust transmon parameters."""
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.3)

    # Initial plot
    (potential_line,) = ax.plot(
        transmon.phi, transmon.potential(), label="Potential", color="black"
    )
    eig_vals, eig_vecs = transmon.hamiltonian()

    wavefunctions = {}
    energy_lines = {}

    for i in range(NUMBER_LEVELS):
        (wavefunctions[i],) = ax.plot(
            transmon.phi, 5 * eig_vecs[:, i] + eig_vals[i], label=f"ψ_{i}"
        )
        energy_lines[i] = ax.axhline(y=eig_vals[i], linestyle="--", lw=1, color="gray")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"Energy/$h$ (GHz)")
    ax.set_ylim([-2, max(eig_vals) + 2])
    ax.set_xlim([-PERIODICITY, PERIODICITY])

    # Sliders
    ax_phi_ext = fig.add_axes([0.1, 0.2, 0.7, 0.04])
    phi_ext_slider = Slider(
        ax_phi_ext, r"$\phi_{ext}$", -2 * np.pi, 2 * np.pi, valinit=transmon.phi_ext
    )

    ax_E_J = fig.add_axes([0.1, 0.15, 0.7, 0.04])
    E_J_slider = Slider(ax_E_J, "$E_J$/h (GHz)", 0.0, 10, valinit=transmon.E_J)

    ax_E_L = fig.add_axes([0.1, 0.1, 0.7, 0.04])
    E_L_slider = Slider(ax_E_L, "$E_L$/h (GHz)", 0.0, 10, valinit=transmon.E_L)

    ax_E_C = fig.add_axes([0.1, 0.05, 0.7, 0.04])
    E_C_slider = Slider(ax_E_C, "$E_C$/h (GHz)", 0.0, 20, valinit=transmon.E_C)

    def update(val):
        """Update plot based on slider values."""
        transmon.phi_ext = phi_ext_slider.val
        transmon.E_J = E_J_slider.val
        transmon.E_L = E_L_slider.val
        transmon.E_C = E_C_slider.val

        # Update potential
        potential_line.set_ydata(transmon.potential())

        # Update eigenvalues and eigenvectors
        eig_vals, eig_vecs = transmon.hamiltonian()
        for i in range(NUMBER_LEVELS):
            wavefunctions[i].set_ydata(5 * eig_vecs[:, i] + eig_vals[i])
            energy_lines[i].set_ydata(eig_vals[i])

        fig.canvas.draw_idle()

    phi_ext_slider.on_changed(update)
    E_J_slider.on_changed(update)
    E_L_slider.on_changed(update)
    E_C_slider.on_changed(update)

    # Reset button
    reset_ax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def reset(event):
        """Reset sliders to initial values."""
        phi_ext_slider.reset()
        E_J_slider.reset()
        E_L_slider.reset()
        E_C_slider.reset()

    reset_button.on_clicked(reset)

    plt.show()


# Example Usage
transmon = TransmonQubit()
plot_transmon_fixed(transmon)  # Fixed plot
plot_transmon_interactive(transmon)  # Interactive plot
