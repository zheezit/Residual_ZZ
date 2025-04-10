import numpy as np
from numpy import linalg as LA
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

import sys
import math as math
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Slider, Button


# np.set_printoptions(threshold=sys.maxsize)

# When you have the good units, you can use these
# e = 1.60217663*10**(-19) # elementary charge in C = coulombs
# hbar = 1.05457182*10**(-34) # m2*kg/ s
# h = 6.62607015*10**(-34) # m2 kg / s

# dont care that much about the units
e = 1
hbar = 1
h = 1
# ------------------------------------------------------------------------------------FLUXONIUM--------------------------------------------------------------------------- #


class Fluxonium:
    def _init_(
        self, C_3, E_J1, phi, phi_ext, gamma
    ):  # C_3 is the capacitanse of the capacitor
        self.E_C = e**2 / (2 * C_3)
        self.E_L = (E_J1 * gamma) / N
        self.delta = phi[3] - phi[2]

    def fluxonium_potential(self):
        return -E_J1 * np.cos(phi - phi_ext) + 1 / 2 * self.E_L * ((phi) ** 2)

    def fluxonium_hamiltonian(self):
        # # make our matrix phi
        Phi = np.zeros((N, N))
        for i in range(N):
            Phi[i][i] = phi[i]

        # # q^2 approximated:
        a = np.ones((1, N - 1))[0]
        b = np.ones((1, N))[0]
        q_2 = np.dot(
            (np.diag(-2 * b, 0) + np.diag(a, -1) + np.diag(a, 1)),
            (-(1)) / (self.delta**2),
        )
        # n_2 = np.square(1/(2*e))*q_2

        # Conductor term: kinetic energy
        C = np.dot(4 * self.E_C, q_2)

        # JJ term: should be a positive diagonal matrix.
        JJ = np.zeros((N, N))
        for i in range(N):
            JJ[i][i] = E_J1 * np.cos(Phi[i][i] - phi_ext)

        # # Inductor term: positiv diagonal matrix
        inductor = np.zeros((N, N))
        for i in range(N):
            inductor[i][i] = 1 / 2 * E_L * (Phi[i][i]) ** 2

        # Define the Hamiltonian
        Hamiltonian = C - JJ + inductor

        eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)
        # print(f"eigenvalues",eig_vals)
        return eig_vals, eig_vec


def fluxonium_potential(E_J, E_L, phi, phi_ext):
    return -E_J * np.cos(phi - phi_ext) + 1 / 2 * E_L * ((phi) ** 2)


def fluxonium_hamiltonian(E_J, E_L, E_C, phi_ext, phi):
    delta = phi[3] - phi[2]
    N = len(phi)
    # # make our matrix phi
    phi_2 = np.zeros((N, N))
    for i in range(N):  # now it is small phi.
        phi_2[i][i] = phi[i]

    # # q^2 approximated:
    a = np.ones((1, N - 1))[0]
    b = np.ones((1, N))[0]
    q_2 = np.dot(
        (np.diag(-2 * b, 0) + np.diag(a, -1) + np.diag(a, 1)), (-(hbar)) / (delta**2)
    )
    # q_2 = np.dot(( np.diag(-2*b,0) + np.diag(a, -1) + np.diag(a, 1)), (-(hbar))/(delta**2) * 1/((2*e)**2))

    # Conductor term: kinetic energy
    C = np.dot(4 * E_C, q_2)

    # JJ term: should be a positive diagonal matrix.
    JJ = np.zeros((N, N))
    for i in range(N):
        JJ[i][i] = E_J * np.cos(phi_2[i][i] - phi_ext)

    # # Inductor term: positiv diagonal matrix
    inductor = np.zeros((N, N))
    for i in range(N):
        inductor[i][i] = 1 / 2 * E_L * (phi_2[i][i]) ** 2

    # Define the Hamiltonian
    Hamiltonian = C - JJ + inductor

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)
    # print(f"eigenvalues",eig_vals)
    return eig_vals, eig_vec
