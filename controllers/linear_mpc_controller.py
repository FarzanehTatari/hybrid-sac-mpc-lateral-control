import numpy as np
import osqp
from scipy import sparse


class MPCController:
    """
    First real linear MPC for lateral vehicle control.

    State:
        x = [e_y, e_psi, v_y, r]

    Input:
        u = delta  (steering angle)

    Notes:
    - Linearized bicycle model around constant longitudinal speed vx
    - Solves a finite-horizon QP with OSQP
    - Enforces steering and steering-rate constraints
    """

    def __init__(self, horizon: int = 15):
        # ---------- Horizon ----------
        self.horizon = horizon

        # ---------- Sampling / model parameters ----------
        self.dt = 0.05
        self.vx = 15.0  # constant longitudinal speed [m/s]

        # Vehicle parameters (matching your current project defaults)
        self.m = 1600.0
        self.Iz = 2500.0
        self.lf = 1.2
        self.lr = 1.6
        self.Cf = 80000.0
        self.Cr = 80000.0

        # ---------- Dimensions ----------
        self.nx = 4   # [e_y, e_psi, v_y, r]
        self.nu = 1   # steering delta

        # ---------- Cost weights ----------
        # Case 2 which was the best in manual tuning
        # self.Q = np.diag([12.0, 6.0, 1.0, 3.0])
        # self.Qf = np.diag([12.0, 6.0, 1.0, 3.0])
        # self.R = np.array([[0.08]])
        # self.Rd = np.array([[2.0]])   # penalty on steering rate change

        # Best case of automated tunng 
        self.Q = np.diag([14.0, 5.0, 1.0, 2.0])
        self.Qf = np.diag([14.0, 5.0, 1.0, 2.0])
        self.R = np.array([[0.06]])
        self.Rd = np.array([[1.5]])   # penalty on steering rate change

        # ---------- Constraints ----------
        self.delta_max = 0.4       # rad  (~23 deg)
        self.delta_rate_max = 0.15 # rad per sample

        # ---------- Internal memory ----------
        self.prev_delta = 0.0

        # Build model and MPC matrices once
        self.A, self.B = self._build_discrete_model()
        self.Phi, self.Gamma = self._build_prediction_matrices()
        self.Qbar, self.Rbar, self.Rdbar = self._build_cost_matrices()
        self.D = self._build_difference_matrix()

        # QP pieces that do not depend on current state x
        self.P = self._build_hessian()
        self.A_cons = self._build_constraint_matrix()

        # Build OSQP solver once; update q, l, u every control step
        self.solver = osqp.OSQP()
        nU = self.horizon * self.nu

        q0 = np.zeros(nU)
        l0 = -1e10 * np.ones(2 * nU)
        u0 =  1e10 * np.ones(2 * nU)

        self.solver.setup(
            P=self.P,
            q=q0,
            A=self.A_cons,
            l=l0,
            u=u0,
            verbose=False,
            warm_start=True
        )

    def reset(self) -> None:
        self.prev_delta = 0.0

    def control(self, x, k: int, **kwargs) -> float:
        """
        Solve the MPC QP for the current state x and return the first steering input.
        """
        x = np.asarray(x, dtype=float).reshape(self.nx, 1)

        # Build linear term q = 2 * f(x)
        q = self._build_linear_term(x)

        # Build constraints that depend on previous steering input
        l, u = self._build_constraint_bounds()

        # Update solver data
        self.solver.update(q=q, l=l, u=u)

        # Solve
        result = self.solver.solve()

        # Fallback if solver fails
        if result.info.status not in ("solved", "solved inaccurate"):
            delta = -0.2 * x[0, 0] - 0.7 * x[1, 0] - 0.05 * x[3, 0]
            delta = np.clip(delta, -self.delta_max, self.delta_max)
        else:
            U_opt = result.x
            delta = float(U_opt[0])

        self.prev_delta = float(delta)
        return float(delta)

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def _build_discrete_model(self):
        """
        Linearized lateral bicycle model around constant vx.
        Continuous-time state:
            x = [e_y, e_psi, v_y, r]
        """
        vx = max(self.vx, 0.1)

        m = self.m
        Iz = self.Iz
        lf = self.lf
        lr = self.lr
        Cf = self.Cf
        Cr = self.Cr

        # Continuous-time matrices
        Ac = np.array([
            [0.0,                         vx,                              1.0, 0.0],
            [0.0,                         0.0,                             0.0, 1.0],
            [0.0,                         0.0,  -(2*Cf + 2*Cr)/(m*vx),   -(vx + (2*Cf*lf - 2*Cr*lr)/(m*vx))],
            [0.0,                         0.0,  -(2*Cf*lf - 2*Cr*lr)/(Iz*vx), -(2*Cf*lf**2 + 2*Cr*lr**2)/(Iz*vx)]
        ])

        Bc = np.array([
            [0.0],
            [0.0],
            [2*Cf / m],
            [2*Cf*lf / Iz]
        ])

        # Simple forward Euler discretization
        A = np.eye(self.nx) + self.dt * Ac
        B = self.dt * Bc

        return A, B

    def _build_prediction_matrices(self):
        """
        Build stacked prediction matrices:
            X = Phi x0 + Gamma U
        where:
            X = [x1; x2; ...; xN]
            U = [u0; u1; ...; u_{N-1}]
        """
        N = self.horizon
        nx = self.nx
        nu = self.nu

        Phi = np.zeros((N * nx, nx))
        Gamma = np.zeros((N * nx, N * nu))

        for i in range(N):
            A_power = np.linalg.matrix_power(self.A, i + 1)
            Phi[i*nx:(i+1)*nx, :] = A_power

            for j in range(i + 1):
                A_ij = np.linalg.matrix_power(self.A, i - j)
                Gamma[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = A_ij @ self.B

        return Phi, Gamma

    def _build_cost_matrices(self):
        """
        Build block-diagonal cost matrices over the horizon.
        """
        N = self.horizon

        Q_blocks = [self.Q for _ in range(N - 1)] + [self.Qf]
        Qbar = sparse.block_diag(Q_blocks, format="csc").toarray()

        R_blocks = [self.R for _ in range(N)]
        Rbar = sparse.block_diag(R_blocks, format="csc").toarray()

        Rd_blocks = [self.Rd for _ in range(N)]
        Rdbar = sparse.block_diag(Rd_blocks, format="csc").toarray()

        return Qbar, Rbar, Rdbar

    def _build_difference_matrix(self):
        """
        Build matrix D such that:
            D U - d = [u0-u_prev, u1-u0, u2-u1, ...]^T
        """
        N = self.horizon
        D = np.zeros((N, N))

        for i in range(N):
            D[i, i] = 1.0
            if i > 0:
                D[i, i - 1] = -1.0

        return D

    # ------------------------------------------------------------------
    # QP building
    # ------------------------------------------------------------------
    def _build_hessian(self):
        """
        Build QP Hessian:
            H = Gamma^T Qbar Gamma + Rbar + D^T Rdbar D

        OSQP solves:
            min 0.5 U^T P U + q^T U
        so we set:
            P = 2H
        """
        H = self.Gamma.T @ self.Qbar @ self.Gamma \
            + self.Rbar \
            + self.D.T @ self.Rdbar @ self.D

        # Ensure symmetry
        H = 0.5 * (H + H.T)

        P = sparse.csc_matrix(2.0 * H)
        return P

    def _build_linear_term(self, x):
        """
        Build linear term q for current state x.

        Cost expansion:
            J = (Phi x + Gamma U)^T Qbar (Phi x + Gamma U)
                + U^T Rbar U
                + (D U - d)^T Rdbar (D U - d)

        Linear term in U:
            2 * [Gamma^T Qbar Phi x - D^T Rdbar d]
        """
        d = np.zeros((self.horizon, 1))
        d[0, 0] = self.prev_delta

        term_state = self.Gamma.T @ self.Qbar @ (self.Phi @ x)
        term_rate = - self.D.T @ self.Rdbar @ d

        q = 2.0 * (term_state + term_rate)
        return q.flatten()

    def _build_constraint_matrix(self):
        """
        Build stacked constraint matrix for:
        1) steering magnitude bounds
        2) steering-rate bounds
        """
        N = self.horizon

        I = sparse.eye(N, format="csc")
        D_sparse = sparse.csc_matrix(self.D)

        # Constraints stacked as:
        #   I * U
        #   D * U
        A_cons = sparse.vstack([I, D_sparse], format="csc")
        return A_cons

    def _build_constraint_bounds(self):
        """
        Bounds for:
        1) -delta_max <= U <= delta_max
        2) -delta_rate_max <= D U - d <= delta_rate_max

        Rewritten as:
            -delta_rate_max + d <= D U <= delta_rate_max + d
        """
        N = self.horizon

        # Steering magnitude bounds
        l_u = -self.delta_max * np.ones(N)
        u_u =  self.delta_max * np.ones(N)

        # Steering rate bounds
        d = np.zeros(N)
        d[0] = self.prev_delta

        l_du = -self.delta_rate_max * np.ones(N) + d
        u_du =  self.delta_rate_max * np.ones(N) + d

        l = np.hstack([l_u, l_du])
        u = np.hstack([u_u, u_du])

        return l, u