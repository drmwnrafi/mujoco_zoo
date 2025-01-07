import sys
import os
import numpy as np
from .plicp_cov import PLICP, KalmanICPCov
from .compute_map import GridMap
from .graph_constructor import PoseGraphOptimization
import g2o
import csv
import scipy
import imageio

class SLAMPipeline:
    def __init__(self, dx_diff=0.05, dx_theta=0.05, lc_mod=10, closure=0.05, xyreso=0.05):
        self.dx_diff = dx_diff
        self.dx_theta = dx_theta
        self.lc_mod = lc_mod
        self.closure = closure
        self.map_res = xyreso

        self.plicp = PLICP()
        self.icp_cov = KalmanICPCov()
        self.optimizer = PoseGraphOptimization()

        self.odom_idx = 0
        self.odom_idx_vertex = []
        self.registered_lasers = []
        self.prev_odom = None
        self.pose = np.eye(3)
        self.vertex_idx = 1
        
        self.occ = GridMap(self.map_res, 0.25, 0.9)
        
        self.max_x = -float('inf')
        self.max_y = -float('inf')
        self.min_x = float('inf')
        self.min_y = float('inf')

        self.images = []
        self.loop_closures = []
        
        self.optimizer.add_vertex(0, g2o.SE2(g2o.Isometry2d(self.pose)), True)

    def add_scan(self, odom, laser):
        if self.odom_idx == 0:
            self.prev_pose = odom.copy()
            self.registered_lasers.append(laser)
            self.odom_idx += 1
            return
        
        dx = odom - self.prev_pose
        
        if np.linalg.norm(dx[:2]) > self.dx_diff or abs(dx[2]) > self.dx_theta:
            ref = np.array(self.registered_lasers[-1])
            state, corresp, src_latest, _ = self.plicp.run(laser.T, ref.T, np.array([[dx[0]], [dx[1]], [dx[2]]]))
            T = self.plicp.homogeneous_transform(state)
            cov_icp = self.icp_cov.run(laser.T, ref.T, self.plicp.rot(state[2]), state[:2], corresp)
            self.pose = np.matmul(self.pose, T)
        
            self.optimizer.add_vertex(self.vertex_idx, g2o.SE2(g2o.Isometry2d(self.pose)))
            information = np.linalg.inv(cov_icp)
            rk = g2o.RobustKernelHuber()
            self.optimizer.add_edge([self.vertex_idx - 1, self.vertex_idx],
                                               g2o.SE2(g2o.Isometry2d(T)),
                                               information, robust_kernel=rk)
        
            if self.vertex_idx > 6 and not self.vertex_idx % self.lc_mod:
                self.detect_loop_closure(laser)
        
            self.prev_pose = odom
            self.odom_idx_vertex.append((self.odom_idx, self.vertex_idx))
            self.registered_lasers.append(laser)
        
            self.vertex_idx += 1
            self.odom_idx += 1
        
            self.optimizer.optimize()
        
            self.closure = False

    def detect_loop_closure(self, laser):
        poses = [self.optimizer.get_pose(idx).to_vector()[0:-1] for idx in range(self.vertex_idx - 1)]
        kd = scipy.spatial.cKDTree(poses)
        x, y, theta = self.optimizer.get_pose(self.vertex_idx).to_vector()
        idxs = kd.query_ball_point(np.array([x, y]), r=1)

        for idx in idxs:
            ref = np.array(self.registered_lasers[idx])
            state, corresp, _, dist = self.plicp.run(laser.T, ref.T)
            if np.mean(dist) < self.closure:
                self.closure = True
                print("Loop Closure !!!")
                T = self.plicp.homogeneous_transform(state)
                information = np.eye(3)
                rk = g2o.RobustKernelDCS()
                self.optimizer.add_edge([self.vertex_idx, idx],
                                                   g2o.SE2(g2o.Isometry2d(T)),
                                                   information, robust_kernel=rk)

    def visualize_map(self):
        traj = np.empty((0, 2))
        point_cloud_list = []
        for idx in range(self.vertex_idx):
            x = self.optimizer.get_pose(idx)
            r = x.to_isometry().R
            t = x.to_isometry().t
            filtered = np.array(self.registered_lasers[idx])
            filtered = filtered[np.linalg.norm(filtered, axis=1) < 50]
            point_cloud_list.append(np.matmul(filtered, r.T) + t)
            traj = np.vstack([traj, x.to_vector()[:2]])

        point_cloud = np.vstack(point_cloud_list)
        min_x, min_y = np.min(point_cloud[:, 0]), np.min(point_cloud[:, 1])
        max_x, max_y = np.max((point_cloud[:, 0] - min_x) // self.map_res), np.max((point_cloud[:, 1] - min_y) // self.map_res)
        map_size = (int(max_x), int(max_y))

        self.occ.clear_map(map_size=map_size)
        for pose_ex, points in zip(traj, point_cloud_list):
            points = points - np.array([min_x, min_y])
            pose_ex = pose_ex[:2] - np.array([min_x, min_y])
            self.occ.update(pose_ex, points)

        map_values = self.occ.mle_map()
        return map_values

# Usage example
# slam = SLAMPipeline()
# for odom, laser_scan in zip(odoms, lasers):
#     slam.add_scan(odom, laser_scan)
#     slam.plot_current_state()
# slam.save_results("../output_graph/slam_results")
