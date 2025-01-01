import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

class Callbacks:
    def __init__(self):
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

    def keyboard(self, window, key, scancode, act, mods, model, data):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)

    def mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        glfw.get_cursor_pos(window)

    def mouse_move(self, window, xpos, ypos, model, scene, cam):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)

        PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)

    def scroll(self, window, xoffset, yoffset, model, scene, cam):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)