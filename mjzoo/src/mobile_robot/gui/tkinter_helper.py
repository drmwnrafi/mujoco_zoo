import customtkinter as ctk
import tkinter as tk
import math
from OpenGL import GL
from pyopengltk import OpenGLFrame
import mujoco as mj
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pynput import keyboard
from mjzoo.modules.icp_slam.slam import SLAMPipeline 

class Joystick(ctk.CTkCanvas):
    def __init__(self, parent, max_distance=50, handle=20, width=None, height=None, **kwargs):
        super().__init__(parent, width=width, height=height, highlightthickness = 0, **kwargs)
        self.configure(bg=parent.cget("fg_color")[1])
        width = ((max_distance * 2) + (handle * 2) + 5 ) if width is None else width
        height = ((max_distance * 2) + (handle * 2) + 5 ) if height is None else height 
        self.max_distance = max_distance
        self.center_x = width // 2
        self.center_y = height // 2
        self.moving_offset_x = self.center_x
        self.moving_offset_y = self.center_y
        self.grab_center = False
        self.dx = 0
        self.dy = 0

        self.base_circle = self.create_aa_circle(
            self.center_x,
            self.center_y,
            self.max_distance,
            math.pi//2,
            fill="#d3d3d3" 
        )

        self.handle_radius = handle
        self.handle = self.create_oval(
            self.center_x - self.handle_radius,
            self.center_y - self.handle_radius,
            self.center_x + self.handle_radius,
            self.center_y + self.handle_radius,
            fill="black"
        )

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<B1-Motion>", self.on_drag)

    def _bound_joystick(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.max_distance:
            scale = self.max_distance / distance
            x = self.center_x + dx * scale
            y = self.center_y + dy * scale
        return x, y

    def joystick_direction(self):
        if not self.grab_center:
            return 0
        self.dx =  ((self.moving_offset_x - self.center_x) / self.max_distance)
        self.dy = -((self.moving_offset_y - self.center_y) / self.max_distance)
        angle = math.degrees(math.atan2(-self.dy, self.dx)) % 360
        distance = min(
            math.sqrt(self.dx**2 + self.dy**2) / self.max_distance, 1.0
        )

    def on_press(self, event):
        self.dx =  ((event.x - self.moving_offset_x) / self.max_distance)
        self.dy = -((event.y - self.moving_offset_y) / self.max_distance)
        if math.sqrt(self.dx**2 + self.dy**2) <= self.handle_radius:
            self.grab_center = True

    def on_release(self, event):
        self.grab_center = False
        self.moving_offset_x = self.center_x
        self.moving_offset_y = self.center_y
        self.update_handle()
        self.dx = 0
        self.dy = 0
        print(self.dx, self.dy)
        
    def on_drag(self, event):
        if self.grab_center:
            self.moving_offset_x, self.moving_offset_y = self._bound_joystick(
                event.x, event.y
            )
            self.update_handle()
            self.joystick_direction()
            print(self.dx, self.dy)

    def update_handle(self):
        self.coords(
            self.handle,
            self.moving_offset_x - self.handle_radius,
            self.moving_offset_y - self.handle_radius,
            self.moving_offset_x + self.handle_radius,
            self.moving_offset_y + self.handle_radius,
        )


class CircularButton(ctk.CTkCanvas):
    def __init__(self, parent, text, radius, fontsize=20, command=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.radius = radius
        self.text = text
        self.original_color = "lightblue"
        self.hover_color = "lightgreen"
        
        self.circle = self.create_aa_circle(radius, radius, radius, fill=self.original_color)
        self.text_id = self.create_text(radius, radius, text=text, font=("Arial", fontsize), fill="black")
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
        if command is None:
            self.bind("<Button-1>", self.on_click)
        else:
            self.bind("<Button-1>", command)
        
    def on_enter(self, event):
        self.itemconfig(self.circle, fill=self.hover_color)

    def on_leave(self, event):
        self.itemconfig(self.circle, fill=self.original_color)
    
    def on_click(self, event):
        print(f"Button {self.text} clicked")
        
def quat2euler(quat):
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    euler = R.from_quat(quat).as_euler('xyz')
    return euler


class MuJoCo(OpenGLFrame):
    def __init__(
        self,
        parent,
        model_path, 
        cam_type=mj.mjtCamera.mjCAMERA_FREE, 
        cam_azimuth=90, 
        cam_elevation=-90.0, 
        cam_distance=10,  
        cam_lookat=np.array([ 0.052671394557840506 , 0.0004384636784280475 , -4.23257624292452 ]), 
        scene_maxgeom=10000, 
        scene_shadow=False,  
        scene_reflection=False, 
        **kwargs
    ):
        super().__init__(parent, **kwargs)
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_x = 0
        self.last_y = 0

        self.model_path = model_path
        self.cam_type = cam_type
        self.cam_azimuth = cam_azimuth
        self.cam_elevation = cam_elevation
        self.cam_distance = cam_distance
        self.cam_lookat = cam_lookat
        self.scene_maxgeom = scene_maxgeom
        self.scene_shadow = scene_shadow
        self.scene_reflection = scene_reflection

        self.robot_pose = None
        self.robot_vel = None
        self.ctrl_inpt = None
        self.laser_scan = None
        self.laser_range = None
        
        self.fps = 0
        self.fps_frame = ctk.CTkCanvas(self, bg="black", highlightthickness=0, width=60, height=20)
        self.fps_frame.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)
        self.fps_label = ctk.CTkLabel(self.fps_frame, text="0 FPS", text_color="white")
        self.fps_label.place(relx=0.5, rely=0.5, anchor="center")
    
        self.bind("<ButtonPress>", self.on_mouse_press)
        self.bind("<ButtonRelease>", self.on_mouse_release)
        self.bind("<Motion>", self.on_mouse_motion)
        self.listener = keyboard.Listener(on_press=self.key_press, on_release=self.key_release)
        self.listener.start()
        self.bind("<BackSpace>", self.reset)
        # self.bind_all("<KeyPress>", self.key_press)
        self.bind("<KeyRelease>", self.key_release)
        self.slam = SLAMPipeline(dx_diff=0.05, 
                                dx_theta=0.1, 
                                lc_mod=20, 
                                closure=0.05,
                                xyreso=0.05)

    def initgl(self):
        self.model = mj.MjModel.from_xml_path(self.model_path)
        self.data = mj.MjData(self.model)
        sweep = 360
        resolution = 1  # degree
        n_lasers = sweep // resolution
        self.angles = np.arange(0, sweep, resolution)
        sensor_types = np.array(self.model.sensor_type[:n_lasers], dtype=int) 
        
        self.lasers_idx = np.where(sensor_types == mj.mjtSensor.mjSENS_RANGEFINDER)[0]
        self.scene = mj.MjvScene(self.model, maxgeom=self.scene_maxgeom)
        self.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = self.scene_shadow
        self.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = self.scene_reflection
        
        self.cam = mj.MjvCamera()
        self.cam.type = self.cam_type
        self.cam.azimuth = self.cam_azimuth
        self.cam.elevation = self.cam_elevation
        self.cam.distance = self.cam_distance
        self.cam.lookat = self.cam_lookat
        
        self.viewport = mj.MjrRect(0, 0, self.width, self.height)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_100.value)
        self.opt = mj.MjvOption()
        
        self.start = time.time()
        self.sim_time = 0
        self.nframes = 0
                        
    def redraw(self):
        time_prev = self.data.time
        while self.data.time - time_prev < 1.0 / 100.0:
            self.sim_time = self.data.time 
            mj.mj_step(self.model, self.data)
            qpos = self.data.qpos
            qvel = self.data.qvel
            ctrl = self.data.ctrl
        
            x, y = qpos[0], qpos[1]
            yaw = quat2euler(qpos[3:7])[2]
            self.robot_pose = np.array([x, y, yaw])
        vx = qvel[0]  
        vy = qvel[1]
        wz = qvel[5]
        self.robot_vel = np.array([vx, vy, wz])
        
        self.ctrl_inpt = ctrl.copy()
        
        lasers_values = self.data.sensordata[self.lasers_idx]
        lasers_values = np.where(lasers_values < 0, np.nan, lasers_values)
        
        x_coords = lasers_values * np.cos(np.radians(self.angles))
        y_coords = lasers_values * np.sin(np.radians(self.angles))
        laser_scan = np.vstack([x_coords, y_coords]).T
        self.laser_scan = laser_scan[~np.isnan(laser_scan).any(axis=1)]
        self.laser_range = lasers_values
        self.slam.add_scan(self.robot_pose, self.laser_scan)

        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        tm = time.time() - self.start
        self.nframes += 1
        self.fps =  self.nframes / tm
        self.fps_label.configure(text=f"{int(self.fps)} FPS")
    
    def on_mouse_press(self, event):
        if 0 <= event.x <= self.width and 0 <= event.y <= self.height:
            if event.num == 1:
                self.button_left = True
            elif event.num == 2:
                self.button_middle = True
            elif event.num == 3:
                self.button_right = True
            self.last_x = event.x
            self.last_y = event.y

    def on_mouse_release(self, event):
        if 0 <= event.x <= self.width and 0 <= event.y <= self.height:
            if event.num == 1:
                self.button_left = False
            elif event.num == 2:
                self.button_middle = False
            elif event.num == 3:
                self.button_right = False

    def on_mouse_motion(self, event):
        if 0 <= event.x <= self.width and 0 <= event.y <= self.height:
            dx = event.x - self.last_x
            dy = event.y - self.last_y
            self.last_x = event.x
            self.last_y = event.y

            width = self.width
            height = self.height

            if self.button_left:
                mj.mjv_moveCamera(
                    self.model, mj.mjtMouse.mjMOUSE_ROTATE_H, dx / width, dy / height, self.scene, self.cam
                )
            elif self.button_middle:
                mj.mjv_moveCamera(
                    self.model, mj.mjtMouse.mjMOUSE_MOVE_H, dx / width, dy / height, self.scene, self.cam
                )
            elif self.button_right:
                mj.mjv_moveCamera(
                    self.model, mj.mjtMouse.mjMOUSE_ZOOM, dx / width, dy / height, self.scene, self.cam
                )

    def move_forward(self,):
        self.data.ctrl = [50, 50]

    def move_backward(self,):
        self.data.ctrl = [-50, -50]

    def move_right(self,):
        self.data.ctrl = [20, -20]

    def move_left(self,):
        self.data.ctrl = [-20, 20]
        
    def stop(self,):
        self.data.ctrl = [0, 0]
        
    def key_press(self, key):
        try:
            if key.char == 'w':  
                self.move_forward()
            elif key.char == 's':  
                self.move_backward()
            elif key.char == 'a':  
                self.move_left()
            elif key.char == 'd': 
                self.move_right()
        except AttributeError:
            self.stop()  

    def key_release(self, key):
        try:
            if key.char in ('w', 'a', 's', 'd'):
                self.stop()
        except AttributeError:
            self.stop() 

    def reset(self, event):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def get_pose(self):
        return self.robot_pose

    def get_velocity(self):
        return self.robot_vel

    def get_control(self):
        return self.ctrl_inpt

    def get_laser(self):
        return self.laser_scan, self.laser_range

    def get_fps(self):
        return self.fps
    
class ShowMap(OpenGLFrame):
    def __init__(self, parent, width, height, **kwargs):
        super().__init__(parent, width=width, height=height, **kwargs)
        self.map_data = None  # To store the map data as a texture

    def initgl(self):
        """Initialize OpenGL settings."""
        glEnable(GL_TEXTURE_2D)  # Enable 2D texture mapping
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color

    def redraw(self):
        """Redraw the OpenGL frame."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen

        if self.map_data is not None:
            self.draw_map()

        self.swap_buffers()  # Swap the buffers to update the display

    def draw_map(self):
        """Draw the map as a texture."""
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_LUMINANCE,
            self.map_data.shape[1],  # Width of the texture
            self.map_data.shape[0],  # Height of the texture
            0,
            GL_LUMINANCE,
            GL_UNSIGNED_BYTE,
            self.map_data,
        )

        # Draw a quad and map the texture onto it
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)  # Bottom-left
        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)   # Bottom-right
        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)    # Top-right
        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)   # Top-left
        glEnd()

    def update_map(self, map_image):
        """Update the map data."""
        self.map_data = map_image
        self.redraw()