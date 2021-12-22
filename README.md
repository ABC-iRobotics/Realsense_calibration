# Realsense calibration
Contains code for calibrating/using RealSense cameras.

## Usage
In order to use the code you need to install the [pyrealsense2](https://pypi.org/project/pyrealsense2/) python package and you need an Intel RealSense camera.

## Realsense controller
The realsense_control.py module defines a RealSenseController class that can be used to initialize the camera and to get RGB and Depth frames as numpy arrays from the camera.

Example:
```python
cam_controller = RealsenseController()
cam_controller.initialize(1920, 1080, 30)  # for resolution of 1920×1080 and FPS of 30
rgb_img_as_np_array, depth_as_np_array = cam_controller.get_frames()
```

# Camera calibration
The camera_calibration.py module defines a CameraCalibration class, that can be used to calibrate the camera and to project 3D points to 2D points onto the image plane or deproject 2D points on image to 3D space.