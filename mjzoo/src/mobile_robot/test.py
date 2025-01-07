from mjzoo.modules.utils import * 
from keyboard_calls import keyboardGLFW
import os
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import mujoco as mj
import time
import os
import tkinter
from OpenGL import GL
import numpy as np
from pyopengltk import OpenGLFrame

class AppOgl(OpenGLFrame):

    def initgl(self):
        """Initalize gl states when the frame is created"""
        glfw.init()
        dirname = os.path.dirname(__file__)
        xml_path = dirname + "/model/scene.xml"
        self.viewport = mj.MjrRect(0, 0, self.width, self.height)
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = 40
        self.cam.elevation = -30
        self.cam.distance = 10
        self.cam.lookat = np.array([0.0, 0.0, 0.0])
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_100.value)
        self.start = time.time()
        self.opt = mj.MjvOption()
        self.nframes = 0
        mj.MjrRect(0, 0, self.width, self.height)
        
    def redraw(self):
        """Render a single frame"""
        mj.mj_step(self.model, self.data)
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(self.viewport, self.scene, self.context)
        tm = time.time() - self.start
        self.nframes += 1
        print("fps",self.nframes / tm, end="\r" )


if __name__ == '__main__':
    root = tkinter.Tk()
    app = AppOgl(root, width=900, height=900)
    glfw.set_key_callback(app, 
                        lambda app, key, scancode, act, mods: 
                            keyboardGLFW(app, 
                                        key, 
                                        scancode, 
                                        act, 
                                        mods, 
                                        app.model, 
                                        app.data))
    glfw.set_cursor_pos_callback(app, 
                                lambda app, xpos, ypos: 
                                    callbacks.mouse_move(app, 
                                                        xpos, 
                                                        ypos, 
                                                        app.model, 
                                                        app.scene, 
                                                        app.cam))
    glfw.set_mouse_button_callback(app, 
                                lambda app, button, act, mods: 
                                    callbacks.mouse_button(app, 
                                                            button, 
                                                            act, 
                                                            mods))
    glfw.set_scroll_callback(app, 
                            lambda app, xoffset, yoffset: 
                                callbacks.scroll(app, 
                                                xoffset, 
                                                yoffset, 
                                                app.model, 
                                                app.scene, 
                                                app.cam))
    print(app)
    app.pack(fill=tkinter.BOTH, expand=tkinter.YES)
    app.animate = 1
    app.after(100, app.printContext)
    app.mainloop()
