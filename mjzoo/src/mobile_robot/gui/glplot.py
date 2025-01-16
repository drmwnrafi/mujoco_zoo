import time
import tkinter
from OpenGL import GL
from pyopengltk import OpenGLFrame
import numpy as np
import math

# Most of the code refers to https://github.com/nickcercone/spectrogram (modernGL code)

class Rect:
    def __init__(self, 
                 x, 
                 y, 
                 width, 
                 height, 
                 color=(0.1, 0.1, 0.1)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        GL.glColor3f(*self.color)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2f(self.x, self.y)
        GL.glVertex2f(self.x + self.width, self.y)
        GL.glVertex2f(self.x + self.width, self.y + self.height)
        GL.glVertex2f(self.x, self.y + self.height)
        GL.glEnd()

class Ruler:
    def __init__(self, 
                 x, 
                 y, 
                 width, 
                 height, 
                 color=(0, 0, 0), 
                 line_length=10, 
                 line_width=1, 
                 gap=50, 
                 direction="horizontal"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.gap = gap
        self.color = color
        self.direction = direction
        self.length = line_length
        self.line_width = line_width

    def draw(self):
        GL.glColor3f(*self.color)
        GL.glLineWidth(self.line_width)
        if self.direction == "horizontal":
            n_lines = (int(self.width) // self.gap)
            GL.glBegin(GL.GL_LINES)
            for i in range(n_lines):
                x = self.x + i * self.gap
                GL.glVertex2f(x, self.y + self.length)
                GL.glVertex2f(x, self.y)
            GL.glEnd()
        elif self.direction == "vertical":
            n_lines = (int(self.height) // self.gap)
            GL.glBegin(GL.GL_LINES)
            for i in range(n_lines):
                y = self.y + self.height - i * self.gap
                GL.glVertex2f(self.x, y)
                GL.glVertex2f(self.x - self.length, y)
            GL.glEnd()
        else:
            raise ValueError("Invalid direction. Use 'horizontal' or 'vertical'.")

class Plotter:
    def __init__(self, 
                 x, 
                 y, 
                 width, 
                 height, 
                 num_data=1, 
                 scaled=False,
                 plot_type="line",  # "line", "wave", or "histogram"
                 line_colors=None, 
                 line_widths=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.num_data = num_data
        self.plot_type = plot_type
        self.scaled = scaled

        if line_colors is None:
            line_colors = np.random.uniform(0, 1, size=(num_data, 3)).tolist() 
        if line_widths is None:
            line_widths = [2] * num_data  

        self.line_colors = line_colors
        self.line_widths = line_widths

        if plot_type == "line": 
            self.samples = [np.full(int(width), np.nan, dtype='f4') for _ in range(num_data)]
            self.vertices = [np.empty((int(width), 2), dtype=np.float32) for _ in range(num_data)]
        
        elif plot_type == "wave":
            self.samples = [np.full(int(width * 2), np.nan, dtype='f4') for _ in range(num_data)]
        
        elif plot_type == "histogram":
            self.samples = [np.zeros(int(width), dtype='f4') for _ in range(num_data)]
        
        self.counter = 0

    def add(self, data, dataset_index=0):
        if dataset_index >= self.num_data:
            raise ValueError(f"Dataset index {dataset_index} is out of range.")
        
        if self.plot_type == "line":
            if self.counter < len(self.samples[dataset_index]):
                self.samples[dataset_index][self.counter] = data
            else:
                self.samples[dataset_index][:-1] = self.samples[dataset_index][1:]
                self.samples[dataset_index][-1] = data
            
            if self.scaled :
                min_val, max_val = np.nanmin(self.samples[dataset_index]), np.nanmax(self.samples[dataset_index])
                if max_val != min_val:
                    self.samples[dataset_index] = 2 * (self.samples[dataset_index] - min_val) / (max_val - min_val) - 1
                else:
                    self.samples[dataset_index] = np.zeros_like(self.samples[dataset_index])
                    
            valid_indices = ~np.isnan(self.samples[dataset_index])
            x_coords = self.x + np.where(valid_indices)[0]
            y_coords = self.y + (self.height / 2) - self.samples[dataset_index][valid_indices] * (self.height / 2)
            self.vertices[dataset_index][:len(x_coords)] = np.column_stack((x_coords, y_coords))
        
        elif self.plot_type == "wave":
            self.samples[dataset_index][:-2] = self.samples[dataset_index][2:]
            self.samples[dataset_index][-2:] = [-data, data]
            
            if self.scaled :
                min_val, max_val = np.nanmin(self.samples[dataset_index]), np.nanmax(self.samples[dataset_index])
                if max_val != min_val:
                    self.samples[dataset_index] = 2 * (self.samples[dataset_index] - min_val) / (max_val - min_val) - 1
                else:
                    self.samples[dataset_index] = np.zeros_like(self.samples[dataset_index])
            
        elif self.plot_type == "histogram":
            self.samples[dataset_index] = np.array(data, dtype='f4') / np.max(data)

        if dataset_index == 0:  
            self.counter += 1
            
    def draw(self):
        for i in range(self.num_data):
            if self.counter == 0:
                continue  
            elif self.plot_type == "line": 
                valid_vertices = self.vertices[i][:self.counter]
                GL.glColor3f(*self.line_colors[i])
                GL.glLineWidth(self.line_widths[i])
                GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
                GL.glVertexPointer(2, GL.GL_FLOAT, 0, valid_vertices)
                GL.glDrawArrays(GL.GL_LINE_STRIP, 0, len(valid_vertices))
                GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

            elif self.plot_type == "wave":
                GL.glColor3f(*self.line_colors[i])
                GL.glLineWidth(self.line_widths[i])
                GL.glBegin(GL.GL_LINE_STRIP)
                for j in range(len(self.samples[i])):
                    GL.glVertex2f(self.x + j / 2, 
                                  self.y + (self.height / 2) - self.samples[i][j] * (self.height / 2))
                GL.glEnd()
            
            elif self.plot_type == "histogram":
                bar_width = self.width / len(self.samples[i])
                GL.glColor3f(*self.line_colors[i])
                
                for j in range(len(self.samples[i])):
                    x1 = self.x + j * bar_width
                    y1 = self.height
                    x2 = x1 + bar_width
                    y2 = self.height - self.samples[i][j] * self.height
                    
                    GL.glBegin(GL.GL_QUADS)
                    GL.glVertex2f(x1, y1)
                    GL.glVertex2f(x2, y1)
                    GL.glVertex2f(x2, y2)
                    GL.glVertex2f(x1, y2)
                    GL.glEnd()
                    
        
class Plot2D(OpenGLFrame):
    def __init__(self, 
                 *args, 
                 num_data=1,
                 plot_type='line', 
                 bg_color=(0, 0.0, 0.0, 1.0),
                 corner_color=(1.0, 0.0, 0.0),
                 corner_radius=10,
                 scaled=False,
                 line_widths=None, 
                 line_colors=None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.nframes = 0
        self.pad = 0
        self.bg_color = bg_color
        self.corner_color = corner_color
        self.corner_radius = corner_radius
        self.num_data = num_data
        self.plot_type=plot_type
        self.line_widths = line_widths
        self.line_colors = line_colors
        self.scaled=scaled
        self.axis_y = None
        self.axis_x = None
        self.plot = None

    def initgl(self):
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(*self.bg_color)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width, self.height, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        
        if self.corner_radius : 
            GL.glEnable(GL.GL_STENCIL_TEST) 

        self.pad = min(self.width // 20, self.height // 20)
        self.draw_axis_lines()
        
        # self.axis_x = Rect(0, 
        #                    self.height - self.pad, 
        #                    self.width, 
        #                    self.pad, 
        #                    (0.5, 0.5, 0.5))
        # self.axis_y = Rect(0, 
        #                    0, 
        #                    self.pad, 
        #                    self.height, 
        #                    (0.5, 0.5, 0.5))
        
        self.plot = Plotter(self.pad, 
                            0, 
                            self.width - self.pad, 
                            self.height - (self.pad + 1), 
                            num_data=self.num_data,
                            scaled=self.scaled,
                            plot_type=self.plot_type,
                            line_widths=self.line_widths,
                            line_colors=self.line_colors)
        
        self.x_ruler = Ruler(self.pad, 
                             self.height - self.pad, 
                             self.width - self.pad, 
                             self.pad, 
                             color=(1, 1, 1),
                             line_length=7,
                             line_width=1.5, 
                             gap=20, 
                             direction="horizontal")
        
        self.y_ruler = Ruler(self.pad, 
                             0, 
                             self.pad, 
                             self.height - self.pad, 
                             color=(1, 1, 1),
                             line_length=7,
                             line_width=1.5,
                             gap=20, 
                             direction="vertical")            
    
    def redraw(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_STENCIL_BUFFER_BIT)
        self.plot.draw()
        # self.axis_y.draw()
        # self.axis_x.draw()
        self.draw_axis_lines()
        self.y_ruler.draw()
        self.x_ruler.draw()
        
        if self.corner_radius :
            GL.glEnable(GL.GL_STENCIL_TEST)
            GL.glStencilFunc(GL.GL_ALWAYS, 1, 0xFF)
            GL.glStencilOp(GL.GL_REPLACE, GL.GL_REPLACE, GL.GL_REPLACE)
            GL.glStencilMask(0xFF)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)  
            self.rounded_rectangle(0, 0, self.width, self.height, self.corner_radius, (0, 0, 0))

            GL.glStencilFunc(GL.GL_NOTEQUAL, 1, 0xFF) 
            GL.glStencilMask(0x00)  
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE) 

            GL.glColor3f(*self.corner_color)  
            GL.glBegin(GL.GL_QUADS)
            GL.glVertex2f(0, 0)
            GL.glVertex2f(self.width, 0)
            GL.glVertex2f(self.width, self.height)
            GL.glVertex2f(0, self.height)
            GL.glEnd()
            GL.glDisable(GL.GL_STENCIL_TEST)
        
        self.nframes += 1
        tm = time.time() - self.start_time
        print(f"fps: {self.nframes / tm:.2f}", end="\r")

    def add_data(self, data, idx):
        if self.plot:
            self.plot.add(data, idx)    
    
    def draw_axis_lines(self):
        # X axis
        GL.glColor3f(1, 1, 1) 
        GL.glLineWidth(1.5)  
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(self.pad, self.height - self.pad)  
        GL.glVertex2f(self.width, self.height - self.pad)  
        GL.glEnd()
        
        # Y axis
        GL.glColor3f(1, 1, 1) 
        GL.glLineWidth(1.5) 
        GL.glBegin(GL.GL_LINES)
        GL.glVertex2f(self.pad, 0) 
        GL.glVertex2f(self.pad, self.height-self.pad)  
        GL.glEnd()
        
    def rounded_rectangle(self, x, y, width, height, radius, color):
        GL.glColor3f(*color)
        GL.glBegin(GL.GL_POLYGON)

        for i in range(0, 91, 10):
            angle = math.radians(i)
            cx = x + width - radius + math.cos(angle) * radius
            cy = y + radius - math.sin(angle) * radius
            GL.glVertex2f(cx, cy)

        for i in range(90, 181, 10):
            angle = math.radians(i)
            cx = x + radius + math.cos(angle) * radius
            cy = y + radius - math.sin(angle) * radius
            GL.glVertex2f(cx, cy)

        for i in range(180, 271, 10):
            angle = math.radians(i)
            cx = x + radius + math.cos(angle) * radius
            cy = y + height - radius - math.sin(angle) * radius
            GL.glVertex2f(cx, cy)

        for i in range(270, 361, 10):
            angle = math.radians(i)
            cx = x + width - radius + math.cos(angle) * radius
            cy = y + height - radius - math.sin(angle) * radius
            GL.glVertex2f(cx, cy)
        GL.glEnd()

if __name__ == '__main__':
    def update_data():
        i = time.time()
        new_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        app.add_data(new_data, idx=0)
        # app.add_data(np.sin(i * 10* np.pi * 0.1), idx=0)
        # app.add_data(np.cos(i * 0.1), idx=1)
        # app.add_data(np.sin(i * 0.1) * np.cos(i * 0.1), idx=2)
        root.after(10, update_data)

    root = tkinter.Tk()
    app = Plot2D(master=root, 
                 num_data=3, 
                 plot_type='histogram', 
                 line_widths= [2]*3,
                 width=300, height=300)
    app.pack(fill=tkinter.BOTH, expand=tkinter.YES)
    update_data()
    app.animate = 1 
    root.mainloop()
