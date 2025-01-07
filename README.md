# A Collection of MuJoCo Models

## How to Use

1. Clone the Repository
    ```bash
    git clone https://github.com/drmwnrafi/mujoco_zoo.git
    ```
2. Create a Python Virtual Environment
    ```bash
    python -m venv mj_zoo
    source mj_zoo/bin/activate  
    # For Windows: mj_zoo\Scripts\activate
    ```
3. Install Dependencies
    ```bash
    cd mujoco_zoo
    pip install -e .
    ```
4.  Run the Simulation
    ```bash
    cd mujoco_zoo
    python mjzoo/src/mobile_robot/main.py 
    # or
    python mjzoo/src/quadcopter/main.py 
    ```

## Model Previews
| **Model**            | **Preview**                                     |
|-----------------------|------------------------------------------------|
| **Simple Quadcopter** | ![Simple Quadcopter](assets/low_poly_f450.png) |
| **F450 Quadcopter**   | ![F450 Quadcopter](assets/f450.png)           |
