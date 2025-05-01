import numpy as np
import qutip as qt
import scipy.linalg as linalg
import scipy.optimize as opt


class Hfluxonium:
    """Hamiltonian-model fluxonium class.

    Used to model analytically the fluxonium Hamiltonian quickly
    and efficiently. Solves in flux basis tridiagonal eigenvalue
    problem for arbitrary Ej, Ec, EL values.

    As long as npoints remains fixed the number phi steps
    considered does not change and it does not recreate the arrays,
    just recomputes the properties

    Returns all properties of interest for the fluxonium.
    """

    def __init__(
        self,
        nlevels: int = 5001,
        Ej: float = None,
        Ec: float = None,
        EL: float = None,
        phib: float = 0.5,
        phic: float = 4,
    ):
        """Generate a a fluxonium model in the phase basis.

        Args:
            nlevels (int): Number of discrete phi steps [-nlevels, nlevels+1]
            Ej (float): Josephson energy of the JJ
            Ec (float): Charging energy of the CPB
            EL (float): Inductive energy of the JJ array
            phib (float): flux bias of the fluxonium
            phic (float): cutoff phase of the phase matrix

        """

        self._nlevels = nlevels
        self._Ej = Ej
        self._Ec = Ec
        self._EL = EL
        self._phib = phib
        self._phic = phic
        self.evals = None
        self.evecs = None
        # Generate the diagonal and offdiagonal components of the Hamiltonian
        self._gen_operators()
        # compute the eigenvectors and eigenvalues of the fluxonium
        # all properties can be derived from these
        self._calc_H()

    def _gen_operators(self):
        """Generate at initialization the number of levels and only recompute
        the size of the problem if nlevels changes."""

        self._diag = np.linspace(-1, 1, self._nlevels)
        # print(f"self._diag = {self._diag}")
        self._off = np.ones(self._nlevels - 1)
        self._ddphi_diag = -2 * np.ones(self._nlevels)
        self._nop = self._gen_nop()

    def _gen_nop(self):
        """Generate the number operator in phase basis"""
        nop = (
            -1j
            * (np.diag(self._off, 1) - np.diag(self._off, -1))
            / (2 * np.pi * self._phic * np.diff(self._diag)[0])
        )
        # print(f"np.diff(self._diag)[0] = {np.diff(self._diag)[0]}")
        # print(f"nop = {nop}")
        # print(f"self._phic = {self._phic}")
        return nop

    def _calc_H(self):
        """Only diagonalize the Hamiltonian if the class is supplied with the
        three mandatory parameters Ej, Ec, EL, but allow for them to not be set
        at initialization."""
        if (self._Ej is None) or (self._Ec is None) or (self._EL is None):
            self.evals = None
            self.evecs = None
        else:
            self._diagonalize_H()

    def _diagonalize_H(self):
        """Diagonalize the fluxonium Hamiltonian using symmetric tridiagonal
        eigensolver for efficient calculation of properties."""
        # Diagional in phi
        phi_op = np.pi * self._diag * self._phic
        # print(f"phi_op = {phi_op}")
        cosphi_op = np.cos(phi_op - 2 * np.pi * self._phib)
        # print(f"cosphi_op = {cosphi_op}")
        delta = np.diff(phi_op)[0]
        # print(f"delta = {delta}")

        # 2nd finite difference matrix has a diagonal component and an off diagonal component
        H_n_sqr_diag = -4 * self._Ec * self._ddphi_diag / (delta**2)
        # print(f"H_n_sqr_diag = {H_n_sqr_diag}")
        H_n_sqr_off = -4 * self._Ec * self._off / (delta**2)
        # print(f"H_n_sqr_off = {H_n_sqr_off}")
        H_q_off = -self._off / (delta**2)
        # Linear inductor component
        H_phi_sq = 0.5 * self._EL * phi_op**2
        # print(f"H_phi_sq = {H_phi_sq}")
        # print(f"phi_op**2 = {phi_op**2}")
        # print(f"phi_op = {phi_op}")
        # JJ component
        H_cosphi = -self._Ej * cosphi_op
        # print(f"H_cosphi = {H_cosphi}")
        # Putting together in tridiagonal matrix
        H_diag = H_n_sqr_diag + H_phi_sq + H_cosphi
        H_off = H_n_sqr_off
        evals, evecs = linalg.eigh_tridiagonal(H_diag, H_off)
        self.evals = np.real(np.array(evals))
        self.evecs = np.array(evecs)

    def evalue_k(self, k: int):
        """Return the eigenvalue of the Hamiltonian for level k.

        Args:
            k (int): Index of the eigenvalue

        Returns:
            float: eigenvalue of the Hamiltonian
        """
        return self.evals[k]

    def evec_k(self, k: int, basis: None | str = None, ncut: int = 11):
        """Return the eigenvector of the fluxonium Hamiltonian for level k.
        The eigenvectors are given in the phase basis

        Args:
            k (int): Index of eigenvector

        Kwargs:
            basis (None|str): which basis states to represent the eigenvector in
                              can be either 'phase' or 'charge'. default to phase
            ncut (int): cuttoff in charge states for the charge representation.

        Returns:
            array: Eigenvector of the |k> level of the fluxonium Hamiltonian
        """
        phi = np.pi * self._phic * self._diag
        evec = self.evecs[:, k]
        nbasis = np.linspace(-ncut, ncut, 1001)
        norm_phi = np.trapezoid(self.evecs[:, k] * self.evecs[:, k].conj(), x=phi)
        evec = evec / np.sqrt(norm_phi)

        if (basis == "phase") or (basis is None):
            return evec
        # Recall that the quadratic potential of the phi operator is now over R
        # this means that the charge operator can no longer be over integers of
        # Cooper pairs

        # It is only due to the Bloch equation that we get this representation of
        # Z (integers) dual to T (circle group). (Pontrayagin duality)

        # For a real value of phase we need to have a duality of R <-> R
        else:
            psi = []
            for i, val in enumerate(phi):
                psi.append(evec[i] * np.exp(-1j * val * nbasis))
            psi = np.array(psi)
            psi_n = np.trapezoid(psi, x=phi, axis=0)
            norm = np.sqrt(np.trapezoid(psi_n * psi_n.conj(), x=nbasis))
            psi_n = psi_n / norm
            return psi_n

    def fij(self, i: int, j: int):
        """Compute the transition energy (or frequency) between states.

        |i> and |j>.

        Args:
            i (int): Index of state |i>
            j (int): Index of state |j>

        Returns:
            float: Eij, the transition energy
        """
        return np.abs(self.evalue_k(i) - self.evalue_k(j))

    def anharm(self):
        """Compute the anharmonicity of the fluxonium.

        Returns:
            float: Anharmonicty defined as E12-E01
        """
        return self.fij(1, 2) - self.fij(0, 1)

    def n_ij(self, i: int, j: int):
        """Compute the value of the number operator for coupling elements
        together in the energy eigen-basis.

        Args:
            i (int): |i> Index of the fluxonium
            j (int): |j> Index of the fluxonium

        Returns:
            float: Matrix element corresponding to the
            number operator in the eigenbasis
            `n_ij = <i|n|j>`
        """
        n_op = self._nop
        # print(f"n_op = {n_op}")
        phi = phi = np.pi * self._phic * self._diag
        # print(f"phi = {phi}")
        dpsi = np.matmul(n_op, self.evec_k(j))
        # print(f"dpsi = {dpsi}")
        n_ij = np.conj(self.evec_k(i)) * dpsi
        # print(f"n_ij = {n_ij}")
        n_ij = np.trapezoid(n_ij, x=phi)
        # print(f"n_ij = {n_ij}")
        return n_ij

    def phi_ij(self, i: int, j: int):
        """Compute the value of the phase operator for driving elements
        together in the energy eigen-basis.

        Args:
            i (int): |i> index of the fluxonium
            j (int): |j> index of the fluxonium

        Returns:
            float: Matrix element corresponding to the phase
            operator in the eigenbasis.
            phi_ij = <i|phi|j>
        """

        phi_op = (
            np.pi * self._phic * self._diag
        )  # Already diagonal so can just multiply vectors
        phi_ij = np.conj(self.evec_k(i)) * phi_op * self.evec_k(j)
        phi_ij = np.trapezoid(phi_ij, x=phi_op)
        return phi_ij

    def h0_to_qutip(self, n_levels: int):
        """Wrapper around Qutip to output the diagonalized Hamiltonian
        truncated up to n levels of the transmon for modeling.

        Args:
            n_levels (int): Truncate up to n levels of the
                            Hamiltonian

        Returns:
            Qobj: Returns a Qutip Qobj for the diagonalized
            Hamiltonian
        """
        ham = np.diag(self.evals[:n_levels] - self.evals[0])
        return qt.Qobj(ham)

    def n_to_qutip(self, n_levels: int, thresh=1e-5):
        """Wrapper around Qutip to output the number operator (charge) for the
        fluxonium Hamiltonian in the energy eigen-basis. Used for computing the
        coupling between other elements in the system.

        Args:
            n_levels (int): Number of energy levels to consider
            thresh (float): Threshold for keeping small values
                            in the number operator i.e `n_{i,i+2}`
                            terms drop off exponentially. If None
                            retain all terms. Defaults to None

        Returns:
            Qobj: Returns a Qutip Qobj corresponding to the
            number operator for defining couplings in the
            energy eigen-basis.
        """
        n_op = 1j * np.zeros((n_levels, n_levels))
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
        return qt.Qobj(n_op)

    def phi_to_qutip(self, n_levels: int, thresh=1e-5):
        """Wrapper around Qutip to output the phase operator for the
        fluxonium Hamiltonian in the energy eigen-basis. Used time modulation
        of the flux

        Args:
            n_levels (int): Number of energy levels to consider
            thresh (float): Threshold for keeping small values
                            in the number operator i.e `n_{i,i+2}`
                            terms drop off exponentially. If None
                            retain all terms. Defaults to None

        Returns:
            Qobj: Returns a Qutip Qobj corresponding to the
            number operator for defining couplings in the
            energy eigen-basis.
        """
        phi_op = 1j * np.zeros((n_levels, n_levels))
        for i in range(n_levels):
            for j in range(n_levels):
                val = self.phi_ij(i, j)
                if thresh is not None:
                    if np.abs(val) < thresh:
                        val = 0
                phi_op[i, j] = val
        return qt.Qobj(phi_op)

    # def params_from_spectrum(self, f01: float, anharm: float, **kwargs):
    #     """Method to work backwards from a desired transmon frequency and
    #     anharmonicty to extract the target Ej and Ec for design and
    #     fabrication. Updates the class to include these Ej and Ec as the new
    #     values for extracting properties.

    #     Args:
    #         f01 (float): Desired qubit frequency
    #         anharm (float): Desired qubit anharmonicity (should be negative)

    #     Keyword Args:
    #         Passed to least_squares

    #     Returns:
    #         (float, float): Ej and Ec of the transmon Hamiltonian
    #         corresponding to the f01 and anharmonicty
    #         of the device
    #     """
    #     # Anharmonicty should be negative for the Transmon
    #     if anharm > 0:
    #         anharm = -anharm

    #     def fun(x):
    #         self.Ej = x[0]
    #         self.Ec = x[1]
    #         # the 10 on the anharmonicity allows faster convergnce, see Minev
    #         return (self.fij(0, 1) - f01)**2 + 10 * (self.anharm() - anharm)**2

    #     # Initial guesses from
    #     # f01 ~ sqrt(8*Ej*Ec) - Ec
    #     #  eta ~ -Ec
    #     x0 = [(f01 - anharm)**2 / (8 * (-anharm)), -anharm]
    #     # can converge slowly if cost function not set up well, or alpha<<freq
    #     ops = dict(bounds=[(0, 0), (x0[0] * 3, x0[1] * 3)],
    #                f_scale=1 / x0[0],
    #                max_nfev=2000)
    #     res = opt.least_squares(fun, x0, **{**ops, **kwargs})
    #     self.Ej, self.Ec = res.x
    #     return res.x

    # def params_from_freq_fixEC(self, f01: float, Ec: float, **kwargs):
    #     """Find transmon Ej given a fixed EC and frequency.

    #     Args:
    #         f01 (float): Desired qubit frequency
    #         Ec (float): Qubit EC (4ECn^2) in same units as f01

    #     Returns:
    #         float: Ej in same units
    #     """

    #     def fun(x):
    #         self.Ej = x[0]
    #         self.Ec = Ec
    #         # the 15 on the anharmonicity allows faster convergnce, see Minev
    #         return (self.fij(0, 1) - f01)**2 + 15 * (self.anharm() - Ec)**2

    #     x0 = [(f01 - Ec)**2 / (8 * (Ec))]
    #     # can converge slowly if cost function not set up well, or alpha<<freq
    #     ops = dict(bounds=[(0,), (x0[0] * 3,)],
    #                f_scale=1 / x0[0],
    #                max_nfev=2000)
    #     res = opt.least_squares(fun, x0, **{**ops, **kwargs})
    #     self.Ej = res.x[0]
    #     self.Ec = Ec
    #     return res.x[0]

    @property
    def nlevels(self):
        """Return the number of levels."""
        return self._nlevels

    @nlevels.setter
    def nlevels(self, value: int):
        """Set the number of levels and recompute the Hamiltonian with the new
        size."""
        self._nlevels = value
        self.__init__(value, self.Ej, self.Ec, self.EL, self._phic)

    @property
    def Ej(self):
        """Returns Ej."""
        return self._Ej

    @Ej.setter
    def Ej(self, value: float):
        """Set Ej and recompute properties."""
        self._Ej = value
        self._calc_H()

    @property
    def Ec(self):
        """Return Ec."""
        return self._Ec

    @Ec.setter
    def Ec(self, value: float):
        """Set Ec and recompute properties."""
        self._Ec = value
        self._calc_H()

    @property
    def EL(self):
        """Return EL."""
        return self._EL

    @EL.setter
    def EL(self, value: float):
        """Set EL and recompute properties."""
        self._EL = value
        self._calc_H()

    @property
    def phib(self):
        """Return phib"""

    @phib.setter
    def phib(self, value: float):
        self._phib = value
        self._calc_H()

    @property
    def phic(self):
        """Return the phi cuttoff"""
        return self._phic

    @phic.setter
    def phic(self, value: float):
        """Set the phi cuttoff and recompute properties"""
        self._phic = value
        self._calc_H()
