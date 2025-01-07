import imgui
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
from mjzoo.modules.utils import * 
from keyboard_calls import keyboardGLFW
import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
from icp_slam.slam import SLAMPipeline 
import threading
import cv2
from scipy.spatial.transform import Rotation as R
import time

class GUI(object):
    def __init__(self, model, data):
        super().__init__()
        self.backgroundColor = (0, 0, 0, 1)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        if not glfw.init():
            print("Could not initialize GLFW")
            exit(1)
        
        self.window = glfw.create_window(900, 900, "Mobile Robot Demo with SLAM", None, None)
        if not self.window:
            glfw.terminate()
            print("Could not create GLFW window")
            exit(1)
        
        glfw.make_context_current(self.window)

        self.context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_100.value)
        gl.glClearColor(*self.backgroundColor)
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        
        callbacks = Callbacks()

        glfw.set_key_callback(self.window, 
                      lambda window, key, scancode, act, mods: 
                          keyboardGLFW(window, key, scancode, act, mods, model, data))
        glfw.set_cursor_pos_callback(self.window, 
                             lambda window, xpos, ypos: 
                                 callbacks.mouse_move(window, xpos, ypos, model, self.scene, self.cam))
        glfw.set_mouse_button_callback(self.window, 
                                lambda window, button, act, mods: 
                                    callbacks.mouse_button(window, button, act, mods))
        glfw.set_scroll_callback(self.window, 
                         lambda window, xoffset, yoffset: 
                             callbacks.scroll(window, xoffset, yoffset, model, self.scene, self.cam))

        self.scene = mj.MjvScene(model, maxgeom=10000)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = 40
        self.cam.elevation = -30
        self.cam.distance = 10
        self.cam.lookat = np.array([0.0, 0.0, 0.0])

        self.opt = mj.MjvOption()

        self.model = model
        self.data = data
        self.txt = ""
        self.f = 0.5
        self.loop()

    def loop(self):
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while (self.data.time - time_prev < 1.0 / 120.0):
                mj.mj_step(self.model, self.data)

            gl.glClearColor(*self.backgroundColor)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            self.impl.process_inputs()
            imgui.new_frame()
            imgui.begin("Custom window", True)
            imgui.text("Hello, world!")

            if imgui.button("OK"):
                print(f"String: {self.txt}")
                print(f"Float: {self.f}")

            _, self.txt = imgui.input_text("A String", self.txt, 256)
            _, self.f = imgui.slider_float("float", self.f, 0.25, 1.5)
            imgui.show_test_window()
            imgui.end()

            imgui.render()
            self.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.impl.shutdown()
        glfw.terminate()

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    xml_path = dirname + "/model/scene.xml"

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    
    gui = GUI(model, data)
