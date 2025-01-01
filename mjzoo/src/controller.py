import numpy as np
from scipy.spatial.transform import Rotation as R

class GeometricControl:
    def __init__(self, kx, kv, kR, komega):
        """Initialize control gains."""
        self.kx = np.array(kx)
        self.kv = np.array(kv)
        self.kR = np.array(kR)
        self.komega = np.array(komega)

    def hat_map(self, v):
        """Hat map for a vector."""
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def vee_map(self, skew):
        """Vee map for a skew-symmetric matrix."""
        return np.array([-skew[1,2],skew[0,2],-skew[0,1]])

    def controller(self, t, t_des, x, x_des, param):
        """Geometric control."""
        g = param["g"]
        m = param["m"]

        px, py, pz = x[:3]
        quat = x[3:7]
        vx, vy, vz = x[7:10]
        omega = x[10:13]

        rotation = R.from_quat(quat, scalar_first=True).as_matrix()

        N = len(x_des[0][0])
        time_idx = min(int(t * (N / t_des[-1])), N - 1)
        print(time_idx)
        xdes, xdes_dot, xdes_ddot, xdes_dddot, xdes_ddddot = x_des[0][:5, time_idx]  # X-axis
        ydes, ydes_dot, ydes_ddot, ydes_dddot, ydes_ddddot = x_des[1][:5, time_idx]  # Y-axis
        zdes, zdes_dot, zdes_ddot, zdes_dddot, zdes_ddddot = x_des[2][:5, time_idx]  # Z-axis
        psi_des, psi_des_dot = x_des[3][:2, time_idx]
        
        ex = np.array([px, py, pz]) - np.array([xdes, ydes, zdes]) # position error
        ev = np.array([vx, vy, vz]) - np.array([xdes_dot, ydes_dot, zdes_dot]) # velocity error
        
        Fd = (-(self.kx * ex) - (self.kv * ev)) + (m * np.array([xdes_ddot, ydes_ddot, zdes_ddot])) + (m * g * np.array([0, 0, 1]))
        zB = rotation[:, 2]
        Fz = np.dot(Fd, zB)
        
        acc_des = np.array([xdes_ddot, ydes_ddot, zdes_ddot + g]).T
        acc = (Fz*zB)/m - g * np.array([0, 0, 1])
        ea = acc - acc_des
        
        jerk = np.array([xdes_dddot, ydes_dddot, zdes_dddot])
        dFd = (-(self.kx * ex) - (self.kv * ea) + jerk)
        dFz = dFd @ zB
    
        if np.linalg.norm(Fd) < 1e-8:
            raise ValueError("Fd is too small")
        zB_des = Fd / np.linalg.norm(Fd)
        xC_des = np.array([np.cos(psi_des), np.sin(psi_des), 0]).T
        zBxC = np.cross(zB_des, xC_des)
        yB_des = zBxC / np.linalg.norm(zBxC)
        xB_des = np.cross(yB_des, zB_des)
        R_des = np.c_[xB_des,yB_des,zB_des]

        u1 = m * np.linalg.norm(acc_des)
        h_omega = (m / u1) * (jerk - (zB_des @ jerk) * zB_des)
        p = -h_omega @ yB_des
        q = h_omega @ xB_des
        r = psi_des_dot * np.array([0, 0, 1]) @ zB_des
        omega_des = np.hstack([p,q,r])

        eR = 0.5 * self.vee_map(R_des.T @ rotation - rotation.T @ R_des) # rotation error
        e_omega = omega - (rotation.T @ R_des @ omega_des)
        M = (-self.kR * eR - self.komega * e_omega)
                
        pos = np.array([px, py, pz])
        pos_des = np.array([xdes, ydes, zdes])
        vel = np.array([vx, vy, vz])
        vel_des = np.array([xdes_dot, ydes_dot, zdes_dot])
        rot = rotation
        rot_des = R_des

        return Fz, M, pos, pos_des, vel, vel_des, omega, omega_des

    def actuator_allocation(self, Fz, M, N_inv):
        """Compute motor inputs from force and moment."""
        return N_inv @ np.array([Fz, *M])