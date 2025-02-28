from utils import * 
from icp_slam.slam import SLAMPipeline 
from keyboard_calls import keyboardGLFW
import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import cv2
from scipy.spatial.transform import Rotation as R
import time

def quat2euler(quat):
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    euler = R.from_quat(quat).as_euler('xyz')
    return euler
    
slam = SLAMPipeline(dx_diff=0.05, 
                    dx_theta=0.1, 
                    lc_mod=20, 
                    closure=0.05,
                    xyreso=0.05)

dirname = os.path.dirname(__file__)
xml_path = dirname + "/model/scene_low.xml"
cam_config = False
callbacks = Callbacks()

model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

cam = mj.MjvCamera()
cam.type = mj.mjtCamera.mjCAMERA_FREE
opt = mj.MjvOption()

glfw.init()
window = glfw.create_window(900, 900, "Mobile Robot Demo with SLAM", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

mj.mjv_defaultFreeCamera(model, cam)
mj.mjv_defaultOption(opt)

scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100.value)

glfw.set_key_callback(window, 
                      lambda window, key, scancode, act, mods: 
                          keyboardGLFW(window, 
                                       key, 
                                       scancode, 
                                       act, 
                                       mods, 
                                       model, 
                                       data))
glfw.set_cursor_pos_callback(window, 
                             lambda window, xpos, ypos: 
                                 callbacks.mouse_move(window, 
                                                      xpos, 
                                                      ypos, 
                                                      model, 
                                                      scene, 
                                                      cam))
glfw.set_mouse_button_callback(window, 
                               lambda window, button, act, mods: 
                                   callbacks.mouse_button(window, 
                                                          button, 
                                                          act, 
                                                          mods))
glfw.set_scroll_callback(window, 
                         lambda window, xoffset, yoffset: 
                             callbacks.scroll(window, 
                                              xoffset, 
                                              yoffset, 
                                              model, 
                                              scene, 
                                              cam))

cam.azimuth = 40
cam.elevation = -30
cam.distance = 10
cam.lookat = np.array([0.0, 0.0, 0.0])

scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = False
num_sensors = model.nsensor

sweep = 360
resolution = 1  # degree
n_lasers = sweep // resolution

angles = np.arange(0, sweep, resolution)
sensor_types = np.array(model.sensor_type[:n_lasers], dtype=int) 
lasers_idx = np.where(sensor_types == mj.mjtSensor.mjSENS_RANGEFINDER)[0]
            
last_time = time.time()

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0 / 120.0):
        mj.mj_step(model, data)
        
        qpos = data.qpos
        
        x, y = qpos[0], qpos[1]
        euler = quat2euler(qpos[3:7])
        yaw = euler[2]
        robot_pose = np.array([x, y, yaw])
        
        lasers_values = data.sensordata[lasers_idx]
        lasers_values = np.where(lasers_values < 0, np.nan, lasers_values)
        
        x_coords = lasers_values * np.cos(np.radians(angles))
        y_coords = lasers_values * np.sin(np.radians(angles))
        laser_scan = np.vstack([x_coords, y_coords]).T
        laser_scan = laser_scan[~np.isnan(laser_scan).any(axis=1)]

        slam.add_scan(robot_pose, laser_scan)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    if cam_config:
        print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ', cam.distance)
        print('cam.lookat = np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

    glfw.swap_buffers(window)
    glfw.poll_events()
    
    now = time.time()
    if (now - last_time >= 3) :
        map_image = slam.visualize_map()
        map_image = cv2.normalize(-map_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        scale_factor = 5
        new_width = int(map_image.shape[1] * scale_factor)
        new_height = int(map_image.shape[0] * scale_factor)
        map_image = cv2.resize(map_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("SLAM Map", map_image)
        cv2.waitKey(1)
        last_time = now

glfw.terminate()

        