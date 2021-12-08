import pyrealsense2 as rs
import numpy as np
import logging

## 
#  @brief Get configuration of the sensors. It is necessary for enable streams.
#  @return configuration of the sensors

def get_config():
    '''
    Get configuration of the sensors. It is necessary for enable streams.

    returns: configuration of the sensors
    '''
    config = rs.config()
    return config


## 
#  @brief Get pipeline of the sensors. It is necessary for enable streams.
#  @return pipeline of the sensors

def get_pipeline():
    '''
    Get pipeline of the sensors. It is necessary for enable streams.

    returns: pipeline of the sensors
    '''
    pipeline = rs.pipeline()
    return pipeline


## 
#  @brief Enable streams (color, depth) in case of SR300
#  @param config is the config for SR300
#  @param rows is the rows for the resolution, 640 is suggested
#  @param cols is the cols for the resolution, 480 is suggested
#  @param framerate is the framerate, 30 is suggested

def enable_stream(config, rows = 640, cols = 480, framerate = 30):
    '''
    Enable streams (color, depth) in case of SR300

    params:
     - config: the config for SR300 (from get_config())
     - rows: the rows for the resolution, 640 is suggested
     - cols: the columns for the resolution, 480 is suggested
     - framerate: the framerate, 30 is suggested
    '''
    if rows <= 0 or cols <= 0:
        raise RuntimeError('Invalid resolution for RealSense initialization')
        return
    if rows >640:
        depth_rows = 640
    else:
        depth_rows = rows
    if cols > 480:
        depth_cols = 480
    else:
        depth_cols = cols
    config.enable_stream(rs.stream.depth, depth_rows, depth_cols, rs.format.z16, framerate)
    config.enable_stream(rs.stream.color, rows, cols, rs.format.bgr8, framerate)


## 
#  @brief Get frames of the sensors
#  @param pipeline is the pipeline
#  @return frames of the sensors

def get_frames(pipeline):
    '''
    Get frames of the sensors

    params:
     - pipeline (from get_pipeline())
    
    returns: frames of the sensors
    '''
    frames = pipeline.wait_for_frames()
    return frames


## 
#  @brief Get depth frames
#  @param frames is the frames coming from the sensor
#  @return the depth frame

def get_depth_frames(frames):
    '''
    Get depth frames

    params:
     - frames: the frames coming from the sensor (from get_frames())
    
    returns: the depth frame
    '''
    depth_frame = frames.get_depth_frame()
    return depth_frame


## 
#  @brief Get color frames
#  @param frames is the frames coming from the sensor
#  @return the color frame

def get_color_frames(frames):
    '''
    Get color frames

    params:
     - frames: the frames coming from the sensor (from get_frames())

    returns: the color frame
    '''
    color_frame = frames.get_color_frame()
    return color_frame


## 
#  @brief Convert frames to nparrays for later usage
#  @param image frames is the image frames coming from the sensor
#  @return images in nparray

def convert_img_to_nparray(image_frames):
    '''
    Convert frames to nparrays for later usage

    params:
     - image_frames: the image frames coming from the sensor (get_depth_frames()/get_color_frames())

    returns: images in nparray

    '''
    img_to_nparray = np.asanyarray(image_frames.get_data())
    return img_to_nparray


class RealsenseController():
    '''
    Class for convenient control of the RealSense camera
    '''

    def __init__(self):
        '''
        Constructor, store camera configuration and pipeline
        '''
        self.camera_config = get_config()
        self.camera_pipeline = get_pipeline()

    def initialize(self, width=640, height=480, frame_rate=30):
        '''
        Initialize the camera streams, start pipeline
        '''
        enable_stream(self.camera_config, width, height, frame_rate)
        self.camera_pipeline.start(self.camera_config)

    def get_frames(self):
        '''
        Get frames from camera as numpy arrays

        returns: (rgb, depth) tuple of numpy arrays containing the RGB frame and the depth frame
        '''
        try:
            frames = get_frames(self.camera_pipeline)
            color_frame = get_color_frames(frames)
            depth_frame = get_depth_frames(frames)

            rgb = convert_img_to_nparray(color_frame)
            depth = convert_img_to_nparray(depth_frame)

            return (rgb, depth)
        
        except:
            logging.error('Could not get frames from RealSense camera, returning None')
            return None
