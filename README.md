# A Collection of MuJoCo Models

This repository is my journey of learning MuJoCo. It also includes the application of Geometric Control for a quadcopter and Levenberg-Marquardt Point-to-Line ICP for mobile robot mapping. 

## Model Previews
| **Model**            | **Preview**                                     |
|-----------------------|------------------------------------------------|
| **Low Quality F450 Quadcopter** | ![Low Quality Quadcopter](assets/low_quality_f450.png) |
| **High Quality F450 Quadcopter**   | ![High Quality Quadcopter](assets/high_quality_f450.png)           |
| **Low Quality Mobile Robot** | ![Low Quality Mobile Robot](assets/low_quality_MR.png) |
| **High Quality Mobile Robot** <br> ( Motor currently not working )   | ![High Quality Mobile Robot](assets/high_quality_MR.png)           |

## How to Use

1. Clone the Repository
    ```bash
    git clone https://github.com/drmwnrafi/mujoco_zoo.git
    ```
2. Create a Python Virtual Environment
    ```bash
    python -m venv mj_zoo
    # For Linux: 
    source mj_zoo/bin/activate  
    # For Windows: 
    mj_zoo\Scripts\activate
    ```
3. Install Dependencies
    ```bash
    cd mujoco_zoo
    pip install -r requirements.txt
    ```
4.  Run the Mujoco GLFW Simulation
    ```bash
    python mjzoo/src/mobile_robot/main.py 
    # or
    python mjzoo/src/quadcopter/main.py 
    ```
    
Run the Mujoco Tkinter GUI (Optional)
    ```bash
    python mjzoo/src/mobile_robot/main_tkinter.py
    ```

<div align="center">
  <a href="https://www.youtube.com/watch?v=9HrDTDOXUvg" target="_blank">
    <img src="https://github.com/drmwnrafi/mujoco_zoo/blob/main/assets/gui.png" alt="MuJoCo Tkinter GUI - Mobile Robot Mapping" />
  </a>
</div>