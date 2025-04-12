from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Slider, Button
import numpy as np
import scipy.linalg as LA
import pandas as pd
import os

import numpy as np
import hvplot.pandas  # for hvplot support
import holoviews as hv
from holoviews import opts
from bokeh.layouts import column
import panel as pn
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import HoverTool

hv.extension("matplotlib")


def plot_fixed(qubit, number_levels=5):
    fig, ax = plt.subplots()
    potential = qubit.potential()

    eig_vals, eig_vecs = qubit.hamiltonian()

    ax.plot(qubit.phi, potential, label="Potential")

    for i in range(number_levels):  # Now using the function argument
        ax.plot(qubit.phi, 5 * eig_vecs[:, i] + eig_vals[i], label=f"ψ_{i}")
        ax.axhline(y=eig_vals[i], linestyle="--", lw=1)

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"Energy/$h$ (GHz)")
    ax.legend()
    plt.show()


def plot_interactive(qubit, number_levels=5, periodicity=2 * np.pi):
    """Generalized interactive plot function for any qubit system."""
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, bottom=0.3)

    (potential_line,) = ax.plot(
        qubit.phi, qubit.potential(), label="Potential", color="black"
    )
    eig_vals, eig_vecs = qubit.hamiltonian()

    wavefunctions = {}
    energy_lines = {}

    for i in range(number_levels):
        (wavefunctions[i],) = ax.plot(
            qubit.phi, 5 * eig_vecs[:, i] + eig_vals[i], label=f"ψ_{i}"
        )
        energy_lines[i] = ax.axhline(y=eig_vals[i], linestyle="--", lw=1, color="gray")

    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"Energy/$h$ (GHz)")
    ax.set_ylim([-2, max(eig_vals) + 2])
    ax.set_xlim([-periodicity, periodicity])

    sliders = []

    def create_slider(label, valmin, valmax, valinit, update_func, position):
        ax_slider = fig.add_axes(position)
        slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit)
        slider.on_changed(update_func)
        sliders.append(slider)
        return slider

    # Define update function
    def update(val):
        """Update plot when sliders change."""
        qubit.E_J = sliders[0].val
        qubit.E_L = sliders[1].val
        qubit.E_C = sliders[2].val
        qubit.phi_ext = sliders[3].val

        potential_line.set_ydata(qubit.potential())
        eig_vals, eig_vecs = qubit.hamiltonian()

        for i in range(number_levels):
            wavefunctions[i].set_ydata(5 * eig_vecs[:, i] + eig_vals[i])
            energy_lines[i].set_ydata(eig_vals[i])

        fig.canvas.draw_idle()

    # Create sliders
    sliders.append(
        create_slider(
            "$E_J$/h (GHz)", 0.0, 10, qubit.E_J, update, [0.1, 0.15, 0.7, 0.04]
        )
    )
    sliders.append(
        create_slider(
            "$E_L$/h (GHz)", 0.0, 10, qubit.E_L, update, [0.1, 0.1, 0.7, 0.04]
        )
    )
    sliders.append(
        create_slider(
            "$E_C$/h (GHz)", 0.0, 20, qubit.E_C, update, [0.1, 0.05, 0.7, 0.04]
        )
    )
    sliders.append(
        create_slider(
            r"$\phi_{ext}$",
            -2 * np.pi,
            2 * np.pi,
            qubit.phi_ext,
            update,
            [0.1, 0.2, 0.7, 0.04],
        )
    )

    # Reset button
    reset_ax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
    reset_button = Button(reset_ax, "Reset")

    def reset(event):
        """Reset sliders to initial values."""
        for slider in sliders:
            slider.reset()

    reset_button.on_clicked(reset)

    plt.show()


# Plotting function using hvplot -------------------------------------------------------------------------------------
def plot_fixed_hvplot(qubit):
    """Plot the fluxonium qubit with fixed parameters using hvplot."""

    eig_vals, eig_vecs = qubit.hamiltonian()

    # Plot the potential
    potential = qubit.potential()
    potential_curve = hv.Curve((qubit.phi, potential), label="Potential").opts(
        xlabel=r"$\phi$", ylabel="Potential Energy", color="black", line_width=2
    )

    # Plot eigenvalues and eigenvectors (energy levels and wavefunctions)
    energy_lines = []
    for i in range(min(len(eig_vals), 5)):  # Limit to first 5 energy levels
        wavefunction = 5 * eig_vecs[:, i] + eig_vals[i]
        energy_lines.append(
            hv.Curve((qubit.phi, wavefunction), label=f"ψ_{i}").opts(
                color=hv.Cycle("Category10")[i], line_width=2
            )
        )
        # Add horizontal lines for the energy levels
        energy_lines.append(
            hv.HLine(eig_vals[i]).opts(
                color=hv.Cycle("Category10")[i], line_dash="dashed"
            )
        )

    # Combine all curves and display the plot
    return (potential_curve * hv.Overlay(energy_lines)).opts(
        opts.Curve(show_grid=True, axiswise=True)
    )


def plot_interactive_hvplot(qubit):
    """Interactive plot with sliders to modify qubit parameters."""

    # Create sliders for parameters
    pn.extension()

    # Define sliders for parameters
    phi_ext_slider = pn.widgets.FloatSlider(
        name=r"$\phi_{\mathrm{ext}}$",
        start=-2 * np.pi,
        end=2 * np.pi,
        value=qubit.phi_ext,
        step=0.1,
    )
    E_J_slider = pn.widgets.FloatSlider(
        name=r"$E_J$", start=0.0, end=10.0, value=qubit.E_J, step=0.1
    )
    E_L_slider = pn.widgets.FloatSlider(
        name=r"$E_L$", start=0.0, end=10.0, value=qubit.E_L, step=0.1
    )
    E_C_slider = pn.widgets.FloatSlider(
        name=r"$E_C$", start=0.0, end=10.0, value=qubit.E_C, step=0.1
    )

    # Define callback function for sliders
    def update_plot():
        qubit.E_J = E_J_slider.value
        qubit.E_L = E_L_slider.value
        qubit.E_C = E_C_slider.value
        qubit.phi_ext = phi_ext_slider.value

        eig_vals, eig_vecs = qubit.hamiltonian()
        potential = qubit.potential()

        potential_curve = hv.Curve((qubit.phi, potential), label="Potential").opts(
            xlabel=r"$\phi$", ylabel="Potential Energy", color="black", line_width=2
        )

        energy_lines = []
        for i in range(min(len(eig_vals), 5)):  # Limit to first 5 energy levels
            wavefunction = 5 * eig_vecs[:, i] + eig_vals[i]
            energy_lines.append(
                hv.Curve((qubit.phi, wavefunction), label=f"ψ_{i}").opts(
                    color=hv.Cycle("Category10")[i], line_width=2
                )
            )
            energy_lines.append(
                hv.HLine(eig_vals[i]).opts(
                    color=hv.Cycle("Category10")[i], line_dash="dashed"
                )
            )

        return (potential_curve * hv.Overlay(energy_lines)).opts(
            opts.Curve(show_grid=True, axiswise=True)
        )

    # Return the panel with the interactive sliders and the plot
    plot = update_plot()
    slider_panel = column(phi_ext_slider, E_J_slider, E_L_slider, E_C_slider)
    return pn.Row(slider_panel, plot)


from bokeh.models import Slider
from bokeh.layouts import column
from bokeh.io import output_notebook, show
import holoviews as hv
import numpy as np
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def interactive_fluxonium_plot(fluxonium, number_levels=5):
    """Create an interactive plot for the Fluxonium qubit."""
    output_notebook()

    # Define the update function
    def update_plot(E_J_val, E_L_val, phi_ext_val):
        fluxonium.E_J = E_J_val
        fluxonium.E_L = E_L_val
        fluxonium.phi_ext = phi_ext_val

        # Compute potential and eigenvalues/wavefunctions
        potential_vals = fluxonium.potential
        eig_vals, eig_vecs = fluxonium.hamiltonian()

        # Plot the potential
        potential_curve = hv.Curve((fluxonium.phi, potential_vals), label="Potential")

        # Plot the energy levels
        energy_lines = []
        for i in range(number_levels):  # Plot the first `number_levels` energy levels
            energy_line = hv.Curve(
                (fluxonium.phi, 5 * eig_vecs[:, i] + eig_vals[i]), label=f"Ψ_{i}"
            )
            energy_lines.append(energy_line)

        # Combine the potential and the energy levels into a single plot
        plot = potential_curve * hv.Overlay(energy_lines)
        plot.opts(
            title="Fluxonium Qubit: Potential and Energy Levels",
            xlabel=r"$\phi$",
            ylabel=r"Energy ($h$ GHz)",
            height=400,
            width=600,
            show_grid=True,
        )

        return plot

    # Create sliders for interactivity
    E_J_slider = Slider(
        start=0.0, end=10.0, value=fluxonium.E_J, step=0.1, title="$E_J$ (GHz)"
    )
    E_L_slider = Slider(
        start=0.0, end=10.0, value=fluxonium.E_L, step=0.1, title="$E_L$ (GHz)"
    )
    phi_ext_slider = Slider(
        start=-2 * np.pi,
        end=2 * np.pi,
        value=fluxonium.phi_ext,
        step=0.01,
        title=r"$\phi_{ext}$",
    )

    # Create the initial plot
    initial_plot = update_plot(fluxonium.E_J, fluxonium.E_L, fluxonium.phi_ext)

    # Update function to refresh the plot based on slider values
    def update_callback(attr, old_value, new_value):
        plot = update_plot(E_J_slider.value, E_L_slider.value, phi_ext_slider.value)
        show(plot)

    # Link the sliders to the update function
    E_J_slider.on_change("value", update_callback)
    E_L_slider.on_change("value", update_callback)
    phi_ext_slider.on_change("value", update_callback)

    # Layout the sliders and plot
    layout = column(initial_plot, E_J_slider, E_L_slider, phi_ext_slider)

    # Display the interactive plot in the notebook
    show(layout)


# Saving functions -------------------------------------------------------------------------------------


def save_plot(filename, fig, dpi=300, format="png"):
    """
    Save a matplotlib figure to the same folder as the script.

    Parameters:
    -----------
    filename : str
        Name of the file without extension
    fig : matplotlib.figure.Figure
        The matplotlib figure to save
    dpi : int, optional
        Resolution in dots per inch, defaults to 300
    format : str, optional
        File format, defaults to "png"
    """
    save_path = os.path.join(os.getcwd(), f"{filename}.{format}")
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Plot saved as: {save_path}")


def save_matrix(
    filename,
    matrix,
    mode="abs",
    format="png",
    x_labels=None,
    y_labels=None,
    title=None,
    cmap="viridis",
):
    """
    Visualize and save a matrix as a heatmap.

    Parameters:
    -----------
    filename : str
        Base name for the saved file
    matrix : numpy.ndarray or QuTiP Qobj
        The matrix to visualize
    mode : str, optional
        Visualization mode: 'abs', 'real', 'imag', or 'angle'
    format : str, optional
        Output file format
    x_labels, y_labels : list, optional
        Custom labels for axes
    title : str, optional
        Custom plot title
    cmap : str, optional
        Matplotlib colormap name
    """
    # Check if input is a QuTiP Qobj and convert to numpy array if needed
    if hasattr(matrix, "full"):  # QuTiP Qobj objects have a 'full' method
        matrix = matrix.full()  # Convert Qobj to numpy array
    elif hasattr(matrix, "toarray"):  # For sparse matrices
        matrix = matrix.toarray()

    print(f"Processing matrix with shape: {matrix.shape}")

    # Process matrix based on mode
    if mode == "abs":
        data = np.abs(matrix)
        cbar_label = "Absolute Value"
    elif mode == "real":
        data = np.real(matrix)
        cbar_label = "Real Part"
    elif mode == "imag":
        data = np.imag(matrix)
        cbar_label = "Imaginary Part"
    elif mode == "angle":
        data = np.angle(matrix)
        cbar_label = "Phase (radians)"
    else:
        raise ValueError("Invalid mode. Use 'abs', 'real', 'imag', or 'angle'.")

    # Create figure correctly
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create custom colormaps with earth tones

    if np.all(data >= 0):  # All positive values
        # Light beige to forest green gradient for positive values
        custom_cmap = LinearSegmentedColormap.from_list(
            "beige_to_green", ["#f5f0e1", "#2c6e49"]  # Light beige to forest green
        )
        norm = None
    else:  # Contains negative values or mixed
        # Terracotta for negative, beige for zero, sage green for positive
        custom_cmap = LinearSegmentedColormap.from_list(
            "terracotta_beige_sage", ["#b84a3b", "#f5f0e1", "#6b9080"]
        )
        vmax = max(abs(np.max(data)), abs(np.min(data)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create heatmap with custom colormap
    heatmap = ax.imshow(data, cmap=custom_cmap, norm=norm, origin="upper")

    # Add a colorbar
    fig.colorbar(heatmap, ax=ax, label=cbar_label)

    # Set tick positions and labels
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    if x_labels:
        ax.set_xticklabels(x_labels)
    if y_labels:
        ax.set_yticklabels(y_labels)

    # Add grid lines
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    # Add labels and title
    ax.set_title(title if title else f"Matrix Plot: {filename} ({mode})")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")

    # Save the figure
    save_path = os.path.join(os.getcwd(), f"{filename}.{format}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved as: {save_path}")
