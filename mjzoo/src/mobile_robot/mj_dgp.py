import os
import numpy as np
import mujoco as mj
import mujoco_viewer as mjv
import dearpygui.dearpygui as dpg
import cv2

dirname = os.path.dirname(__file__)
xml_path = dirname + "/model/scene.xml"

model = mj.MjModel.from_xml_path(xml_path)
data_mj = mj.MjData(model)

# Laser setup
sweep = 360
resolution = 4  # degree
n_lasers = sweep // resolution
angles = np.arange(0, sweep, resolution)
sensor_types = np.array(model.sensor_type[:n_lasers], dtype=int)
lasers_idx = np.where(sensor_types == mj.mjtSensor.mjSENS_RANGEFINDER)[0]

dpg.create_context()
dpg.create_viewport(title='MAP-IT', width=800, height=800)
dpg.setup_dearpygui()

def update_texture(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_rgb = np.asfarray(image_rgb, dtype='f')
    texture_data = np.true_divide(image_rgb, 255.0)
    return texture_data.ravel()

with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        600, 
        600,
        np.zeros(600 * 600 * 3), 
        tag="texture_tag",
        format=dpg.mvFormat_Float_rgb
    )

with dpg.window(label="Example Window", tag="Primary Window"):
    dpg.add_image("texture_tag")

viewer = mjv.MujocoViewer(model, data_mj, 'offscreen', width=600, height=600)
viewer.cam.azimuth = 90.19999999999993
viewer.cam.elevation = -89.0
viewer.cam.distance = 5.542061060684448
viewer.cam.lookat = np.array([0.0, 0.0, 0.0])
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
cv2.namedWindow("Mujoco Render", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mujoco Render", 800, 800)
while dpg.is_dearpygui_running():
    time_prev = data_mj.time
    while (data_mj.time - time_prev < 1.0 / 120.0):
        mj.mj_step(model, data_mj)
        lasers_values = data_mj.sensordata[lasers_idx]
        lasers_values = np.where(lasers_values < 0, np.nan, lasers_values)

    image = viewer.read_pixels(camid=-1)
    print(f"{image[0][:10]} \n {image[1][:10]} \n {image[2][:10]}")
    img_inp = update_texture(image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    dpg.set_value("texture_tag", img_inp)
    dpg.render_dearpygui_frame()
    cv2.imshow("Mujoco Render", image_bgr)

    # Break if 'q' is pressed in OpenCV
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()

dpg.destroy_context()
