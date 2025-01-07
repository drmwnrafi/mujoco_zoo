import numpy as np
import scipy
from scipy.optimize import least_squares

class PLICP:
    def __init__(self, max_iterations = 80, outliers_threshold = 1.0, lambda_init = 0.003, error_convergence = 0.001):
      self.max_iterations = max_iterations
      self.outliers_threshold = outliers_threshold
      self.error_convergence = error_convergence
      self.lambda_init = lambda_init

    def rot(self, theta):
      return np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]).squeeze()

    def lorentzian(self, error):
      return np.log(1 + error**2/self.outliers_threshold)

    def HuberLoss(self, error):
        norm_error = np.abs(error)
        expression = norm_error <= self.outliers_threshold
        weights = np.ones_like(error)
        weights[~expression] = self.outliers_threshold / norm_error[~expression]
        return weights

    def CauchyLoss(self, error):
      return 1 / (1 + (error / self.outliers_threshold)**2)

    def compute_normals(self, points, step=1):
      shifted_plus = np.roll(points, -step, axis=1)
      shifted_minus = np.roll(points, step, axis=1)
      dx = shifted_plus[0] - shifted_minus[0]
      dy = shifted_plus[1] - shifted_minus[1]
      normals = np.stack([-dy, dx], axis=0)
      norms = np.linalg.norm(normals, axis=0)
      normals = normals / norms
      normals[:, 0] = np.array([0, 0])
      normals[:, -1] = np.array([0, 0])
      return normals.T[:, None, :]

    def compute_error(self, state, source, target, correspondences, target_normals, kernel_func):
      target_normals_corr = target_normals[correspondences]
      error = np.einsum('ijk,ki->ij', target_normals_corr, (self.rot(state[2]) @ source + state[0:2].reshape((2, 1)) - target[:, correspondences]))
      weight = kernel_func(error).squeeze()
      normals_x = target_normals_corr[:, 0, 0]
      normals_y = target_normals_corr[:, 0, 1]
      third_component = normals_x * ((-source[0, :] * np.sin(state[2])) + (source[1, :] * -np.cos(state[2]))) + \
                        normals_y * ((source[0, :] * np.cos(state[2])) + (source[1, :] * -np.sin(state[2])))
      J = np.stack((normals_x, normals_y, third_component), axis=-1)
      J = J[:, np.newaxis, :]
      H = np.sum(weight * np.einsum("ijk,kjl->ilk", J.T, J,), axis=2)
      g = np.sum(weight * np.einsum("ijk,kj->ijk", J.T, error,), axis=2)
      chi = np.sum(error**2)
      return H, g, chi

    def homogeneous_transform(self, state):
      homogeneous_matrix = np.eye(3)
      homogeneous_matrix[:2, :2] = self.rot(state[2])
      homogeneous_matrix[:2, 2] = state[0:2].flatten()
      return homogeneous_matrix

    def run(self, source, target, init_state=None):
        r'''
        Running Iterative Closest Point with point-to-line error metric (PLICP). The algorithm finds the optimal
        transformation given 2 point clouds (target and source) that minimize the error function :

        \underset{R_{\theta}, t}{\operatorname{arg\,min}} \sum_i \| n_i \cdot (R_{\theta}p_i+t-q_j)\|^2

        where R_{\theta} is a rotation matrix with theta angle, p_i is a point of source point cloud,
        q_j is a point of target point cloud.

        Algorithm consist of the following steps:
        1. Find the correspondences for each point in source w.r.t target point by using KDTree.
        2. Apply distance filtering for correspondences.
        3. Calculate hessian, gradient, chi^2, and error.
        4. Solve least square problem using Levenberg-Marquardt.
        5. Algorithm iterates until convergence.

        Parameters
        ----------
        source : array-like with shape (2, N)
            Point cloud that wants to match
        target : array-like with shape (2, N)
            Point cloud as a reference for source to match.
        init_state (optional) : array with shape (3, 1)
            Initial state to transform source points.
        '''
        state = np.zeros((3, 1))
        if init_state is not None:
            state = init_state
        source_latest = source
        target_normals = self.compute_normals(target)
        nn = scipy.spatial.cKDTree(target.T)
        prev_chi = 0
        lambda_param = self.lambda_init
        for i in range(self.max_iterations):
            dist, correspondences = nn.query(source_latest.T, k=1)
            mask = dist <= 0.08
            H, g, chi = self.compute_error(state, source[:, mask], target, correspondences[mask], target_normals, self.HuberLoss)
            H_lm = H + lambda_param * np.diag(H)
            delta_state = np.linalg.lstsq(H_lm, -g, rcond=None)[0]
            new_state = state + delta_state
            new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
            new_source_latest = self.rot(new_state[2]) @ source + new_state[0:2]
            _, _, new_chi = self.compute_error(new_state, source[:, mask], target, correspondences[mask], target_normals, self.HuberLoss)
            if new_chi < chi:
                state = new_state
                source_latest = new_source_latest
                chi = new_chi
                lambda_param /= 10
            else:
                lambda_param *= 10
            criteria = np.linalg.norm(prev_chi - chi)
            if criteria < self.error_convergence:
              break
            prev_chi = chi
        return state, correspondences, source_latest, dist

class KalmanICPCov():
    def compute_noise(self, source, reference, rot, trans, corresp):
        """
        Compute noise using the formula:
        \sigma_m^2 = \frac{1}{N} \Sigma_{i=1}^N ||p_{s,i} - Rp_{r,\phi(i)} - t||^2
        """
        operation = (source - np.einsum("ij,jk->ik", rot, reference[:, corresp]) - trans)**2
        noise = np.mean(np.linalg.norm(operation, axis=0))
        return noise

    def compute_normal(self, source, reference, corresp):
        """
        Compute normal using the formula:
        n = \frac{p_{s,i}-p_{r,\phi(i)}}{||p_{s,i}-p_{r,\phi(i)}||}
        """
        diff = source - reference[:, corresp]
        diff_norm = np.linalg.norm(diff, axis=0)
        diff_norm[diff_norm < 1e-6] = 1e-6
        return diff / diff_norm

    def run(self, source, reference, rot, trans, corresp):
        P = np.eye(3) * 1e6
        noise = self.compute_noise(source, reference, rot, trans, corresp)
        n = self.compute_normal(source, reference, corresp)
        for i, j in enumerate(corresp):
            v = rot @ reference[:, j]
            d_r = np.array([[0, -1], [1, 0]]) @ v
            n_x, n_y = n[:, i]
            H = np.array([[n_x, n_y, d_r @ n[:, i]]])
            S = H @ P @ H.T + noise
            K = P @ H.T @ np.linalg.inv(S)
            P = (np.identity(3) - K @ H) @ P
        return P