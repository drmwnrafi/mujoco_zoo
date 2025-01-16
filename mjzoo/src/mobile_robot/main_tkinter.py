import os
import numpy as np
import mujoco as mj
import time
import customtkinter as ctk
import tkinter as tk
from gui.tkinter_helper import *
from gui import glplot
from tkinter import ttk
import threading
import queue
from dataclasses import dataclass
from PIL import Image, ImageTk
import cv2

@dataclass
class AppConfig:
    dark_blue: str = "#2D2F33"
    black: str = "#242424"
    active_tab_color: str = "#9B9B9F"
    text_color: str = "#ffffff"
    tab_color: str = "#B4BCC1"
    custom_font: tuple = ("Roboto", 10)
    green: str = "#19FF99"
    red: str = "#FF1999"
    blue: str = "#1999E1"
    dark_blue_rgb: tuple = (45/225, 47/255, 51/225, 1)
    green_byte = (0.09803921568627451, 1, 0.6)
    red_byte = (1, 0.09803921568627451, 0.6)
    blue_byte = (0.09803921568627451, 0.6, 0.8823529411764706)

    def __post_init__(self):
       self.title_font = ctk.CTkFont("Roboto", size=30, weight="bold")
       self.label_font = ctk.CTkFont("Roboto", size=15, weight="bold")

class App:
    def __init__(self, root, config: AppConfig):
        self.root = root
        self.config = config
        self.setup_appearance()
        self.create_layout()
        self.create_column1()
        self.create_column2()

    def setup_appearance(self):
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme(os.path.dirname(__file__) + "/gui/themes/rime.json")
        
        self.style = ttk.Style()
        self.style.configure("TNotebook",
                padding=(5, 5, 5, 5),
                background=self.config.dark_blue,
                borderwidth=0,
                relief="flat",
                font=self.config.custom_font)

        self.style.configure("TNotebook.Tab",
                        focuscolor=self.config.dark_blue, 
                        background=self.config.tab_color,
                        padding=(16, 6, 16, 6),
                        height=24,
                        borderwidth=0,
                        relief="groove",
                        font=self.config.custom_font)

        self.style.map("TNotebook.Tab",
                background=[("active", self.config.black), ("selected", self.config.black)],
                foreground=[("active", self.config.text_color), ("selected", self.config.text_color)])
    
    def create_layout(self):
        self.root.grid_columnconfigure(0, weight=1000)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)   

    def create_column1(self):
        self.mujoco_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.mujoco_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.mujoco_screen = MuJoCo(self.mujoco_frame, model_path=xml_path, width=500, height=500)
        self.mujoco_screen.pack(side=ctk.LEFT, padx=10, pady=10, fill=ctk.BOTH, expand=True)
        self.mujoco_screen.animate = 1

    def create_column2(self):
        self.info_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.info_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.info_frame.grid_columnconfigure(0, weight=1)
        self.info_frame.grid_rowconfigure(0, weight=1) 
        self.info_frame.grid_rowconfigure(1, weight=2)
        
        self.setup_general_info()
        
        self.info_color = self.info_frame.cget("bg_color")[1]
        
        self.tabview = ttk.Notebook(self.info_frame)
        
        graph_tab = tk.Frame(self.tabview, bg=self.info_color)
        self.setup_graph_tab(graph_tab)
        self.tabview.add(graph_tab, text="Graph")
        
        map_tab = tk.Frame(self.tabview, bg=self.info_color)
        self.setup_map_tab(map_tab)
        self.tabview.add(map_tab, text="Map")
        
        self.tabview.grid(padx=10, pady=(0, 5), row=1, column=0, sticky="nsew")
    
        
    def setup_general_info(self):
        self.general_info = ctk.CTkFrame(self.info_frame, corner_radius=15, fg_color=self.config.black)
        self.general_info.grid(padx=10, pady=(5, 5), row=0, column=0, sticky="nsew")
        general_label = ctk.CTkLabel(self.general_info,
                                     text="DASHBOARD", 
                                     font=self.config.title_font, 
                                     text_color="white",
                                     fg_color=self.config.black)
        general_label.pack(pady=7, padx=4)

        sim_frame = ctk.CTkFrame(self.general_info, corner_radius=10, fg_color="gray20")
        sim_frame.pack(pady=5, padx=10, fill="x")

        pose_label = ctk.CTkLabel(sim_frame,
                                  text="Pose",
                                  font=self.config.label_font,
                                  text_color="white",
                                  fg_color="gray20",
                                  pady=5)
        pose_label.pack(pady=3, padx=(10,5), side=ctk.LEFT)
        
        self.x_label = ctk.CTkLabel(sim_frame,
                               text="X : 0.00",
                               font=self.config.label_font,
                               text_color="#ECEFF4",
                               fg_color=self.config.black,
                               corner_radius=5)
        self.x_label.pack(pady=5, padx=5, side=ctk.LEFT)

        self.y_label = ctk.CTkLabel(sim_frame,
                               text="Y : 0.00",
                               font=self.config.label_font,
                               text_color="#ECEFF4",
                               fg_color=self.config.black,
                               corner_radius=5)
        self.y_label.pack(pady=5, padx=5, side=ctk.LEFT)
        
        self.theta_label = ctk.CTkLabel(sim_frame,
                                   text="θ : 0.00",
                                   font=self.config.label_font,
                                   text_color="#ECEFF4",
                                   fg_color=self.config.black,
                                   corner_radius=5)
        self.theta_label.pack(pady=5, padx=5, side=ctk.LEFT)

        self.sim_time_label = ctk.CTkLabel(sim_frame,
                                   text="0.00 s",
                                   font=self.config.label_font,
                                   text_color="#ECEFF4",
                                   fg_color=self.config.black,
                                   corner_radius=5)
        self.sim_time_label.pack(pady=5, padx=5, side=ctk.RIGHT)
        
        sim_label = ctk.CTkLabel(sim_frame,
                                      text="Simulation Time",
                                      font=self.config.label_font,
                                      text_color="#ECEFF4",
                                      fg_color=self.config.black,
                                      corner_radius=5)
        sim_label.pack(pady=5, padx=5, side=ctk.RIGHT)
        
        callbacks_frame = ctk.CTkFrame(self.general_info, fg_color="transparent")
        callbacks_frame.pack(pady=(5, 10), padx=10, fill="x")

        callbacks_frame.grid_columnconfigure(0, weight=1) 
        callbacks_frame.grid_columnconfigure(1, weight=1)

        mouse_callback_frame = ctk.CTkFrame(callbacks_frame, corner_radius=10, fg_color="gray20")
        mouse_callback_frame.grid(row=0, column=0, padx=(0, 5), pady=2, sticky="nesw")

        mouse_callback_label = ctk.CTkLabel(
            mouse_callback_frame,
            text="Mouse Callbacks",
            font=self.config.label_font,
            text_color="white",
            fg_color="gray20"
        )
        mouse_callback_label.pack(padx=5, pady=1, fill=ctk.X)
        self.callbacks_label(mouse_callback_frame, "Right Click", "Camera Zoom")
        self.callbacks_label(mouse_callback_frame, "Wheel Click", "Move Horizontal")
        self.callbacks_label(mouse_callback_frame, "Left  Click", "Camera Rotate Horizontal")

        keyboard_callback_frame = ctk.CTkFrame(callbacks_frame, corner_radius=10, fg_color="gray20")
        keyboard_callback_frame.grid(row=0, column=1, padx=(5, 0), pady=2, sticky="nesw")

        keyboard_callback_label = ctk.CTkLabel(
            keyboard_callback_frame,
            text="Keyboard Callbacks",
            font=self.config.label_font,
            text_color="white",
            fg_color="gray20"
        )
        keyboard_callback_label.pack(padx=5, pady=1, fill=ctk.X)
        self.callbacks_label(keyboard_callback_frame, "W", "Move Forward")
        self.callbacks_label(keyboard_callback_frame, "A", "Turn Left")
        self.callbacks_label(keyboard_callback_frame, "S", "Move Backward")
        self.callbacks_label(keyboard_callback_frame, "D", "Turn Right") 
        self.callbacks_label(keyboard_callback_frame, "Backspace", "Reset") 
    
    def callbacks_label(self, parent, key, action):
        label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        label_frame.pack(padx=5, pady=(0, 5), fill=ctk.X)
        key_label = ctk.CTkLabel(label_frame,
                                text=key,
                                font=("Roboto", 12),
                                text_color=self.config.black,
                                fg_color=self.config.tab_color,
                                corner_radius=5, 
                                anchor="center")
        key_label.pack(fill=ctk.X, side=ctk.RIGHT, padx=5)  
        action_label = ctk.CTkLabel(label_frame,
                                    text=action,
                                    font=("Roboto", 12),
                                    text_color="white",
                                    fg_color=self.config.black,
                                    corner_radius=5,
                                    anchor="center")
        action_label.pack(fill=ctk.X, padx=5)
        
    def setup_graph_tab(self, graph_tab):
        graph_tab.grid_columnconfigure((0, 1), weight=1)
        graph_tab.grid_rowconfigure((0, 1), weight=1)
        
        # Position Frame
        self.pos_frame = ctk.CTkFrame(graph_tab, fg_color=self.config.dark_blue, corner_radius=10)
        self.pos_frame.grid(row=0, column=0, padx=(10, 5), pady=5, sticky="nesw")
        self.custom_label(self.pos_frame, "Position", [self.config.blue, self.config.green], ["Y", "X"])
        
        # Velocity Frame
        self.vel_frame = ctk.CTkFrame(graph_tab, fg_color=self.config.dark_blue, corner_radius=10)
        self.vel_frame.grid(row=0, column=1, padx=(5, 10), pady=5, sticky="nesw")
        self.custom_label(self.vel_frame, "Velocity", [self.config.blue, self.config.green], ["Y", "X"])
        
        # Lidar Frame
        self.lidar_frame = ctk.CTkFrame(graph_tab, fg_color=self.config.dark_blue, corner_radius=10)
        self.lidar_frame.grid(row=1, column=0, padx=(10, 5), pady=5, sticky="nesw")
        self.custom_label(self.lidar_frame, "Lidar", [self.config.red], ["Laser Range"])
        
        # Control Input Frame
        self.ctrl_frame = ctk.CTkFrame(graph_tab, fg_color=self.config.dark_blue, corner_radius=10)
        self.ctrl_frame.grid(row=1, column=1, padx=(5, 10), pady=5, sticky="nesw")
        self.custom_label(self.ctrl_frame, "Control Input", [self.config.blue, self.config.green], ["L", "R"])

    def setup_map_tab(self, map_tab):
        map_frame = ctk.CTkFrame(map_tab, corner_radius=15, bg_color=self.info_color)
        map_frame.pack(padx=20, pady=20, fill=ctk.BOTH, expand=True)
        self.custom_label(map_frame, "Occupancy Grid Map", ["black", "white", "gray"], ["OCCUPIED", "FREE", "UNEXPLORED"])
        self.map = tk.Canvas(map_frame)
        self.map.place(relx=0.5, rely=0.5, anchor="center")
        
    def custom_label(self, parent, main_text, colors, texts):
        label_frame = ctk.CTkFrame(parent, fg_color="transparent")
        label_frame.pack(side=ctk.TOP, pady=4, fill=ctk.X, padx=10)
        
        main_label = ctk.CTkLabel(label_frame, text=main_text, font=self.config.label_font, fg_color="transparent")
        main_label.pack(side=ctk.LEFT, padx=(0, 10))
        
        for color, text in zip(colors, texts):
            if text:
                if color == "black" :
                    text_color =  "white"
                else :
                    text_color =  "black"
                box_frame = ctk.CTkFrame(label_frame, width=12, height=10, fg_color=color, corner_radius=5)
                box_frame.pack(side=ctk.RIGHT, padx=(0, 5))
                box_label = ctk.CTkLabel(box_frame, text=text, text_color = text_color,fg_color=color, corner_radius=5)
                box_label.pack(pady=0.5, padx=4)

def update_info(plots, app):
    while True:
        try :
            sim_time = app.mujoco_screen.sim_time
            robot_pose = app.mujoco_screen.get_pose()  
            velx, vely, velyaw = app.mujoco_screen.get_velocity()  
            L, R = app.mujoco_screen.get_control()  / 50
            scan, lrange = app.mujoco_screen.get_laser() 
            
            app.x_label.configure(text=f"X : {robot_pose[0]:.2f}")
            app.y_label.configure(text=f"Y : {robot_pose[1]:.2f}")
            app.theta_label.configure(text=f"θ : {np.rad2deg(robot_pose[2]):.2f}°")
            app.sim_time_label.configure(text=f"{sim_time:.2f} s")
            
            data_queue.put((plots[0], robot_pose[0], 0))
            data_queue.put((plots[0], robot_pose[1], 1))
            data_queue.put((plots[1], velx, 0))  
            data_queue.put((plots[1], vely, 1))  
            data_queue.put((plots[2], L, 0))  
            data_queue.put((plots[2], R, 1))
            data_queue.put((plots[3], lrange, 0)) 
        except Exception as e:
            pass
        time.sleep(0.001)
        
    
def process_queue():
    try: 
        while not data_queue.empty():
            plot, value, idx = data_queue.get()
            plot.add_data(value, idx)
    except Exception as e:
            pass
    root.after(100, process_queue)

    
def update_map(app):
    while not stop_map_update.is_set():
        try:
            map_image = app.mujoco_screen.slam.visualize_map()
            map_image = cv2.normalize(-map_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            img = Image.fromarray(map_image)
            w, h = map_image.shape[1] * 5, map_image.shape[0] * 5
            img_resized = img.resize((w, h), Image.NEAREST)
            img_tk = ImageTk.PhotoImage(image=img_resized)
            app.map.create_image(0, 0, image=img_tk, anchor="nw")
            app.map.configure(width=w, height=h)
            app.map.image = img_tk
        except Exception as e:
            print(f"Error in update_map: {e}")
        time.sleep(1)
        
def on_tab_changed(event):
    global map_update_thread, stop_map_update
    
    selected_tab = app.tabview.tab(app.tabview.select(), "text")
    
    if selected_tab == "Map":
        if map_update_thread is None or not map_update_thread.is_alive():
            stop_map_update.clear()
            map_update_thread = threading.Thread(target=update_map, args=(app,), daemon=True)
            map_update_thread.start()
    else:
        stop_map_update.set()
        if map_update_thread is not None:
            map_update_thread.join(timeout=0.1)
            map_update_thread = None
            
if __name__ == "__main__":       
    def on_close():
        root.destroy()
        os._exit(0) 

    data_queue = queue.Queue()
    map_update_thread = None
    stop_map_update = threading.Event()
    dirname = os.path.dirname(__file__)
    xml_path = os.path.join(dirname, "model", "scene_low.xml")
    
    root = ctk.CTk()
    config = AppConfig()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}")
    root.title("MuJoCo Simulation")
    app = App(root, config=config)
    app.tabview.bind("<<NotebookTabChanged>>", on_tab_changed)
    
    plot_frames = [app.pos_frame, app.vel_frame, app.ctrl_frame, app.lidar_frame]
    plots = []

    for i, frame in enumerate(plot_frames):
        color = [config.green_byte, config.blue_byte]
        if i == 3:
            color = [config.red_byte]
            g_plot = glplot.Plot2D(frame, num_data=1, line_colors=color, plot_type="histogram")
        elif i == 2:
            g_plot = glplot.Plot2D(frame, scaled=False, line_colors=color, num_data=2)
        else:
            g_plot = glplot.Plot2D(frame, scaled=True, line_colors=color, num_data=2)
        g_plot.bg_color = config.dark_blue_rgb
        g_plot.corner_radius = None
        g_plot.pack(padx=3, pady=(0, 5), fill=ctk.BOTH, expand=True)
        g_plot.animate = 1
        plots.append(g_plot)
        
    threading.Thread(target=update_info, args=(plots, app), daemon=True).start()
    root.after(100, process_queue)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()