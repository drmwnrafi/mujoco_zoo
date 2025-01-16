import mujoco as mj
from mujoco.glfw import glfw

def keyboardGLFW(window, key, scancode, act, mods, model, data):
    if act == glfw.PRESS or act == glfw.REPEAT:
        if key == glfw.KEY_W:  
            data.ctrl[0] = 50  
            data.ctrl[1] = 50  
        elif key == glfw.KEY_S:
            data.ctrl[0] = -50  
            data.ctrl[1] = -50  
        elif key == glfw.KEY_D:
            data.ctrl[0] = 20  
            data.ctrl[1] = -20  
        elif key == glfw.KEY_A: 
            data.ctrl[0] = -20 
            data.ctrl[1] = 20  
        elif key == glfw.KEY_BACKSPACE: 
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)

    if act == glfw.RELEASE:
        if key in [glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D]:
            data.ctrl[0] = 0
            data.ctrl[1] = 0