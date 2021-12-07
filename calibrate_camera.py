import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CameraCalibration():
    '''
    Class for calibrating and using the Realsense cameras
    '''

    def __init__(self, configs_folder):
        self.configs_folder = configs_folder
        if os.path.exists(configs_folder):
            try:
                files = ['cam_mtx.npy', 'dist.npy', 'new_cam_matrix.npy', 'inverse_new_cam_matrix.npy']
                contets = []
                for fi in files:
                    with open(os.path.join(configs_folder, fi), 'rb') as f:
                        contets.append(np.load(f))
                self.cam_mtx, self.dist, self.newcam_mtx, self.inverse_new_cam_matrix = contets
            except:
                self.cam_mtx, self.dist, self.newcam_mtx, self.inverse_new_cam_matrix = [None]*4
        else:
            os.makedirs(configs_folder)
            self.cam_mtx, self.dist, self.newcam_mtx, self.inverse_new_cam_matrix = [None]*4
        self.log = logging.getLogger(__name__)

    def set_loglevel(self, level=logging.WARNING):
        '''
        Set loglevel of camera calibration class

        arguments:
         - level (enum (logging.levels)): Treshold level for logging (if smaller than WARNING will be set to DEBUG automatically)
        '''
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)
        if level >= logging.WARNING:
            self.log.setLevel(logging.WARNING)
        else:
            self.log.setLevel(logging.DEBUG)
    
    def calibration(self, chessboard_shape, chessboard_square_size, calibration_images_folder):
        '''
        Calibrate the camera

        arguments:
         - chessboard_shape (tuple): Tuple containing the shape of the internal grid of corners of the chessboard (e.g. (4,6))
         - chessboard_square_size (double): Side length of one square in the chessboard (in millimeters)
         - calibration_images_folder (string/path): Path to folder containing the images for calibration
        '''
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Create 3D points of chessboard in the coordinate frame fixed to the corner of the chessboard
        obj_points = np.zeros((chessboard_shape[0]*chessboard_shape[1],3), np.float32)
        obj_points[:,:2] = np.mgrid[0:chessboard_shape[0],0:chessboard_shape[1]].T.reshape(-1,2)*chessboard_square_size

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        image_files = os.listdir(calibration_images_folder)

        # Get imagepoints by finding the corners of chessboard pattern on the images (the corresponding object points are the same for all images (the chessboard's shape does not change))
        for image_file in image_files:
            img = cv2.imread(os.path.join(calibration_images_folder, image_file))

            if not img is None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

                if ret == True:
                    objpoints.append(obj_points)
                    cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners)

        if objpoints and imgpoints and not gray is None:
            ret, cam_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        self.log.info('Calibration successful: ' + str(ret))
        self.log.info("Camera Matrix")
        self.log.info(cam_mtx)
        np.save(os.path.join(self.configs_folder,'cam_mtx.npy'), cam_mtx)
        self.cam_mtx = cam_mtx

        self.log.info("Distortion Coeff")
        self.log.info(dist)
        np.save(os.path.join(self.configs_folder,'dist.npy'), dist)
        self.dist = dist

        newcam_mtx, roi=cv2.getOptimalNewCameraMatrix(cam_mtx, dist, img.shape[:2], 1, img.shape[:2])

        self.log.info("Region of Interest")
        self.log.info(roi)

        self.log.info("New Camera Matrix")
        self.log.info(newcam_mtx)
        np.save(os.path.join(self.configs_folder,'new_cam_matrix.npy'), newcam_mtx)
        self.newcam_mtx = newcam_mtx

        inverse = np.linalg.inv(newcam_mtx)
        self.log.info("Inverse New Camera Matrix")
        self.log.info(inverse)
        np.save(os.path.join(self.configs_folder,'inverse_new_cam_matrix.npy'), inverse)
        self.inverse_new_cam_matrix = inverse


        if self.log.level < logging.WARNING:
            img = cv2.imread('../../oe_logo_demo/calib_images/00000.png')

            undst = cv2.undistort(img, cam_mtx, dist, None, newcam_mtx)

            x,y,w,h = roi

            plt.imshow(cv2.cvtColor(np.concatenate((img, undst), axis=0), cv2.COLOR_BGR2RGB))
            plt.show()


            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, dist)
                error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error

            self.log.info("Total error: " +  str(mean_error/len(objpoints)))

    def project_3d_to_2d(self, points_3d, frame_transform):
        '''
        Project 3D points onto the 2D image

        arguments:
         - points_3d (list): list of x,y,z, values, in the form of [[x,y,z]...]
         - frame_transform (tuple): tuple of form ((x,y,z), (rx,ry,rz)), the translation and rotation of the frame (relative to the camera frame) in which the "3d_points" are given

        returns:
         - points_2d (list): list of 2D points or None if camera is not calibrated
        '''
        if not self.cam_mtx is None and not self.dist is None:

            points_3d = np.array(points_3d).astype(np.float32)

            rx, ry, rz = frame_transform[1]
            rvec = np.array([[rx], [ry], [rz]]).astype(np.float)

            x, y, z = frame_transform[0]
            tvec = np.array([[x], [y], [z]]).astype(np.float)

            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.cam_mtx, self.dist)
            return (points_2d)
        else:
            logging.warn('The camera matrix and the distrotion values are empty, either the reading of the config files failed or the camera is not calibrated!')
            return(None)

    def deproject_2d_to_3d(self, img_points, z):
        '''
        Deproject 2D image points to 3D space supposing that the 3D points are located on a plane parallel to the image plane in a distance of "z" from the camera optical center

        arguments:
         - img_points (list): List of 2D points in image space in the form of [(u,v),(u,v)]
         - z (double): Distance of 3D plane from camera center

        returns:
         - points_3d (list): list of 3D points or None if camera is not calibrated
        '''
        if not self.inverse_new_cam_matrix is None and not self.dist is None:
            points_3d = []
            for img_point in img_points:
                uv1 = np.array([img_point[0],img_point[1],1])

                xyz = self.inverse_new_cam_matrix.dot(uv1*z)

                x_tilde = xyz[0]/xyz[2]
                y_tilde = xyz[1]/xyz[2]

                r_sqr = x_tilde**2 + y_tilde**2

                k1,k2,p1,p2,k3 = self.dist[0]

                x_2_tilde = x_tilde*(1+k1*r_sqr+k2*r_sqr**2+k3*r_sqr**3)+2*p1*x_tilde*y_tilde+p2*(r_sqr+2*x_tilde**2)
                y_2_tilde = y_tilde*(1+k1*r_sqr+k2*r_sqr**2+k3*r_sqr**3)+2*p2*x_tilde*y_tilde+p1*(r_sqr+2*y_tilde**2)

                points_3d.append((x_2_tilde*xyz[2], y_2_tilde*xyz[2], xyz[2]))
            return(points_3d)
        else:
            logging.warn('The camera new matrix and the distrotion values are empty, either the reading of the config files failed or the camera is not calibrated!')
            return(None)

if __name__=='__main__':

    chessboard_shape = (4,6)
    chessboard_square_size = 35  # in mm
    calibration_images_folder = '../../oe_logo_demo/calib_images'

    cam = CameraCalibration('../../oe_logo_demo/calib_results')
    cam.set_loglevel(logging.WARNING)

    # Project 3D points to 2D
    points_3d = [[10,0,0]]
    print(cam.project_3d_to_2d(points_3d, ((0,0,200),(0,0,0))))

    # Calibrate camera
    cam.calibration(chessboard_shape,chessboard_square_size,calibration_images_folder)

    # Deproject 2D points to 3D
    print(cam.deproject_2d_to_3d([[0,0]], 430))
