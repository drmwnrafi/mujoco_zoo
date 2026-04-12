from utils import * 
import numpy as np
import os
import mujoco as mj
from mujoco.glfw import glfw
from controller import GeometricControl
from differential_flatness import TrajectoryOptimizer
import matplotlib.pyplot as plt
        
def render_waypoints(scene, waypoints):
    for waypoint in waypoints:
        mj.mjv_initGeom(
            geom=scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_BOX,
            size=0.2 * np.array([1, 1, 1], dtype=np.float64),
            pos=np.array(waypoint[:3], dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 0.5], dtype=np.float32)
        )
        scene.ngeom += 1

        yaw = waypoint[3] 
        direction = np.array([np.cos(yaw), np.sin(yaw), 0], dtype=np.float64)
        arrow_start = np.array(waypoint[:3], dtype=np.float64)
        arrow_end = arrow_start + 0.5 * direction  # Scale direction length

        mj.mjv_initGeom(
            geom=scene.geoms[scene.ngeom],
            type=mj.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([0.05, 0, 0], dtype=np.float64),  # Capsule radius
            pos=(arrow_start + arrow_end) / 2,
            mat=np.eye(3).flatten(),
            rgba=np.array([0, 0, 1, 1], dtype=np.float32)  # Blue arrow
        )
        scene.ngeom += 1
        
def render_trajectory(scene, trajectory_points):
    for i in range(len(trajectory_points) - 1):
        point1 = trajectory_points[i]
        point2 = trajectory_points[i + 1]

        direction = point2 - point1
        length = np.linalg.norm(direction)

        if length > 0:
            direction /= length
            radius = 0.05
            geom = scene.geoms[scene.ngeom]
            mj.mjv_initGeom(geom, 
                            mj.mjtGeom.mjGEOM_CAPSULE, 
                            np.zeros(3), 
                            np.zeros(3), 
                            np.zeros(9), 
                            np.array([0.0, 1.0, 1.0, 0.5], 
                                     dtype=np.float32))
            mj.mjv_connector(geom, 
                             mj.mjtGeom.mjGEOM_CAPSULE, 
                             radius, 
                             point1, 
                             point2)
            scene.ngeom += 1

dirname = os.path.dirname(__file__)
xml_path = dirname + "/model/scene_low_poly.xml"

callbacks = Callbacks()

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)
cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_FREE
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(900, 900, "Geometric Control Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultFreeCamera(model, cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100.value)

# glfw.set_key_callback(window, lambda window, key, scancode, act, mods: callbacks.keyboard(window, key, scancode, act, mods, model, data))
glfw.set_cursor_pos_callback(window, lambda window, xpos, ypos: callbacks.mouse_move(window, xpos, ypos, model, scene, cam))
glfw.set_mouse_button_callback(window, lambda window, button, act, mods: callbacks.mouse_button(window, button, act, mods))
glfw.set_scroll_callback(window, lambda window, xoffset, yoffset: callbacks.scroll(window, xoffset, yoffset, model, scene, cam))

cam.azimuth = 40
cam.elevation = -30
cam.distance = 40
cam.lookat = np.array([0.0, 0.0, 0.0])

param = {
    "m": max(model.body_mass),
    "J": np.diag(model.body_inertia[1, :]),
    "g": 9.81,  
}

kx     = [10, 10, 10] 
kv     = [5, 5, 5]
kR     = [25, 25, 25] 
komega = [6, 6, 6]  # control gains

waypoints = np.array([
    [0, 0, 0, 0],
    [0, 0, 5, 0],
    [5, 0, 6, 0],
    [0, -5, 7, 0],
    [-5, 0, 8, 0],
    [0, 5, 9, 0],
])
n_coeffs = [8, 8, 8, 4]  
derivatives = [4, 4, 4, 2] 
times = [0, 3, 8, 12, 16, 22]

optimizer = TrajectoryOptimizer(n_coeffs, derivatives, times)
states, coeff = optimizer.generate_trajectory(waypoints, num_points=400)

controller = GeometricControl(kx, kv, kR, komega)

x_traj = states[0][0,:]
y_traj = states[1][0,:]
z_traj = states[2][0,:]

trajectory_points = np.vstack([x_traj, y_traj, z_traj]).T

kf = 1
km = 0.0201
dx = 0.79450535
dy = 0.7945588
# dx = 0.14
# dy = 0.18
# c = 0.0201
N = np.array([[kf, kf, kf, kf],
              [-kf*dy, 0, 0, -kf*dy],
              [0, 0, -kf*dx, -kf*dx],
              [-km, km, -km, km]])
N_inv = np.linalg.inv(N)

motor_names = ["motor1", "motor2", "motor3", "motor4"]
motor_ids = [model.actuator(name).id for name in motor_names]

simulation_time = []
actual_positions = []
actual_velocities = []
desired_positions = []
desired_velocities = []
actual_rot = []
actual_omega = []
desired_rot = []
desired_omega = []


while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0 / 60.0):
        qpos = data.qpos
        qvel = data.qvel

        position = qpos[:3]
        quaternion = qpos[3:7]
        velocity = qvel[:3]
        angular_velocity = qvel[3:6]
        
        x = np.concatenate((position, quaternion, velocity, angular_velocity))

        t = data.time
        Fz, M, pos, pos_des, vel, vel_des, omega, omega_des = controller.controller(t, times, x, states, param)
        motor_inputs = controller.actuator_allocation(Fz, M, N_inv)
        
        for motor_id, force in zip(motor_ids, motor_inputs):
            data.ctrl[motor_id] = force  
            
        simulation_time.append(t)
        actual_positions.append(pos)
        actual_velocities.append(vel)
        desired_positions.append(pos_des)
        desired_velocities.append(vel_des)
        actual_omega.append(omega)
        desired_omega.append(omega_des)

        mj.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    render_waypoints(scene, waypoints)
    # render_trajectory(scene, trajectory_points[::5])
    mj.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()