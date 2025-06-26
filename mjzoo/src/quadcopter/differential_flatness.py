import numpy as np
from scipy.special import perm
from cvxopt import matrix, solvers

class TrajectoryOptimizer:
    def __init__(self, n_coeffs:list, derivatives:list, times:list):
        """
        Initialize the TrajectoryOptimizer class.

        Parameters:
        - n_coeffs (list[int]): List of polynomial coefficients for each segment.
        - derivatives (list[int]): List of derivative orders for each segment.
        - times (list[float]): List of time durations for each segment.
        """
        self.n_coeffs = n_coeffs
        self.derivatives = derivatives
        self.times = times
        self.T = np.ediff1d(times)

    @staticmethod
    def poly_coeff(n, d, t):
        """
        Compute the polynomial coefficients for the derivative constraints.
        """
        assert n > 0 and d >= 0
        D = n - 1 - np.arange(n)
        j = np.arange(d)[:, None]
        factors = np.where(D - j >= 0, D - j, 0)
        prod = np.prod(factors, axis=0)
        exponents = np.maximum(D - d, 0)
        cc = prod * t**exponents
        return cc[::-1].astype(float)

    def hessian(self, n, d, t):
        """
        Compute the cost function matrix Q for a polynomial trajectory optimization problem.
        """
        num_t = len(t)
        Q_size = num_t * n
        Qi = np.zeros((Q_size, Q_size))

        for i in range(num_t):
            start_idx = i * n
            end_idx = start_idx + n

            for l in range(n):
                for k in range(n):
                    if l >= d and k >= d:
                        pow_term = l + k - 2 * d + 1
                        product = perm(l, d) * perm(k, d)
                        Qi[start_idx + l, start_idx + k] = 2 * product * (t[i] ** pow_term / pow_term)

        return Qi

    def q_block(self):
        """
        Generate a block-diagonal matrix of Hessian matrices for polynomial trajectory optimization.
        """
        size = sum(self.n_coeffs) * len(self.T)
        Q_block = np.zeros((size, size))
        cum_idx = 0

        for i, (order, d) in enumerate(zip(self.n_coeffs, self.derivatives)):
            Qi = self.hessian(order, d, self.T)
            block_size = Qi.shape[0]
            Q_block[cum_idx:cum_idx + block_size, cum_idx:cum_idx + block_size] = Qi
            cum_idx += block_size

        return Q_block

    def constraint(self):
        """
        Generate a constraint matrix for quadratic programming.
        """
        n_T = len(self.T)
        n_segments = n_T - 1
        n_axes = len(self.n_coeffs)
        n_constraints = sum((n_T * 2 + n_segments * d) for d in self.derivatives)
        n_coeffs_total = sum(self.n_coeffs) * n_T
        A = np.zeros((n_constraints, n_coeffs_total))
        f = np.zeros(n_coeffs_total)

        start_idx = 0
        
        # Start & end position constraints
        for i in range(n_axes):
            for j in range(n_T):
                idx = i * n_T + j
                end_idx = start_idx + self.n_coeffs[i]
                A[idx, start_idx: end_idx] = self.poly_coeff(self.n_coeffs[i], 0, 0)
                A[n_axes * n_T + idx, start_idx: end_idx] = self.poly_coeff(self.n_coeffs[i], 0, self.T[j])
                start_idx = end_idx

        # Continuous derivatives constraints
        num_pos_constraints = n_axes * n_T * 2
        cumulative_offset = 0

        for j in range(1, max(self.derivatives) + 1):
            valid_axes = np.where(j <= np.array(self.derivatives))[0]
            n_axes = len(valid_axes)
            for k in range(n_segments):
                for i in valid_axes:
                    row_index = num_pos_constraints + cumulative_offset + (i * n_segments + k)
                    start_col = sum(self.n_coeffs[:i]) * n_T + k * self.n_coeffs[i]
                    end_col = start_col + self.n_coeffs[i] * 2
                    coeffs_left = self.poly_coeff(self.n_coeffs[i], j, self.T[k])
                    coeffs_right = -self.poly_coeff(self.n_coeffs[i], j, 0)
                    A[row_index, start_col: end_col] = np.concatenate([coeffs_left, coeffs_right])        
            cumulative_offset += n_axes * n_segments

        return A, f    

    def target(self, waypoint):
        """
        Generate the target vector for the constraints.
        """
        if waypoint.ndim == 1:
            waypoint = np.expand_dims(waypoint, axis=1)
        n_wp, n_axes = waypoint.shape
        n_T = len(self.T)
        n_segments = n_T - 1
        n_constraints = sum((n_T * 2 + n_segments * d) for d in self.derivatives)
        b = np.zeros(n_constraints)

        for axis in range(n_axes):
            b[axis * n_T : (axis + 1) * n_T] = waypoint[:-1, axis]
            b[n_axes * n_T + axis * n_T : n_axes * n_T + (axis + 1) * n_T] = waypoint[1:, axis]
        return b

    def generate_trajectory(self, waypoint, num_points=100):
        """
        Solve the optimization problem and generate the trajectory.
        """
        Q = matrix(self.q_block())
        A, f = self.constraint()
        f = matrix(f)
        A = matrix(A)
        b = matrix(self.target(waypoint))
        sol = solvers.qp(Q, f, None, None, A, b)
        coeff = list(sol['x'])

        N = num_points
        t = np.linspace(self.times[0], self.times[-1], N)
        states = []

        for axis in range(len(self.n_coeffs)):
            d_states = np.zeros((self.derivatives[axis] + 1, N))
            for i in range(N):
                j = np.nonzero(t[i] <= self.times)[0][0] - 1
                j = max(j, 0)
                ti = t[i] - self.times[j]
                start_idx = sum(self.n_coeffs[:axis]) * len(self.T) + self.n_coeffs[axis] * j
                end_idx = start_idx + self.n_coeffs[axis]
                c = np.flip(coeff[start_idx:end_idx])
                current_coeff = c
                for d in range(self.derivatives[axis] + 1):
                    d_states[d, i] = np.polyval(current_coeff, ti)
                    current_coeff = np.polyder(current_coeff)
            states.append(d_states)
        return states, coeff

    def get_yaw(self, vel):
        curr_heading = vel/np.linalg.norm(vel)
        prev_heading = self.heading
        cosine = max(-1,min(np.dot(prev_heading, curr_heading),1))
        dyaw = np.arccos(cosine)
        norm_v = np.cross(prev_heading,curr_heading)
        self.yaw += np.sign(norm_v)*dyaw

        if self.yaw > np.pi: self.yaw -= 2*np.pi
        if self.yaw < -np.pi: self.yaw += 2*np.pi

        self.heading = curr_heading
        yawdot = max(-30,min(dyaw/0.005,30))
        return self.yaw,yawdot
    
    def solve(self, waypoint, t):
        """
        Evaluate the trajectory at a specific time t for the given waypoint.
        """
        Q = matrix(self.q_block())
        A, f = self.constraint()
        f = matrix(f)
        A = matrix(A)
        b = matrix(self.target(waypoint))
        sol = solvers.qp(Q, f, None, None, A, b)
        coeff = list(sol['x'])

        states = []

        for axis in range(len(self.n_coeffs)):
            d_states = np.zeros(self.derivatives[axis] + 1)
            j = np.nonzero(t <= self.times)[0][0] - 1
            j = max(j, 0)
            ti = t - self.times[j]
            start_idx = sum(self.n_coeffs[:axis]) * len(self.T) + self.n_coeffs[axis] * j
            end_idx = start_idx + self.n_coeffs[axis]
            c = np.flip(coeff[start_idx:end_idx])
            current_coeff = c
            for d in range(self.derivatives[axis] + 1):
                d_states[d] = np.polyval(current_coeff, ti)
                current_coeff = np.polyder(current_coeff)
            states.append(d_states)
        return states


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Define 2D waypoints (x, y)
    waypoints = np.array([
        [0, 0],
        [0, 1500],
        [1500, 1500],
    ])

    n_coeffs = [8, 8]
    derivatives = [2, 2]
    times = [0, 1, 2]

    optimizer = TrajectoryOptimizer(n_coeffs, derivatives, times)

    states, coeffs = optimizer.generate_trajectory(waypoints, num_points=500)
    t = np.linspace(times[0], times[-1], 500)

    x = states[0][0, :]  # X positions
    y = states[1][0, :]  # Y positions
    vel_x = states[0][1, :]  # X velocities
    vel_y = states[1][1, :]  # Y velocities
    
    # Calculate heading angles from velocities
    yaw = np.arctan2(vel_y, vel_x)
    
    # Create main plot
    plt.figure(figsize=(12, 6))
    
    # Trajectory plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x, y, label='Trajectory')
    ax1.plot(waypoints[:, 0], waypoints[:, 1], 'ro', label='Waypoints')
    

    ax1.set_title('Trajectory with Orientation')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    # Add orientation arrows
    skip = 20  # Plot every 20th arrow
    for i in range(0, len(x), skip):
        dx = np.cos(yaw[i]) * 50  # Arrow size scaling
        dy = np.sin(yaw[i]) * 50
        ax1.arrow(x[i], y[i], dx, dy, 
                head_width=30, head_length=50, 
                fc='darkorange', ec='navy')
    
    plt.tight_layout()
    plt.show()