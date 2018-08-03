import cv2
import numpy as np


class Attention:

    def predict_intersection(self, yaw, pitch, roll, tx, ty, face_height, face_width, scale):

        depth_est = 0 # in ft
        pixels_to_feet = scale

        # assumed scale: 25 pixels to 1 ft
        # use width (in pixels) to estimate depth
        if face_width < 60:
            depth_est = pixels_to_feet * 18
        elif face_width < 100:
            depth_est = pixels_to_feet * 14
        elif face_width < 150:
            depth_est = pixels_to_feet * 10
        else:
            depth_est = pixels_to_feet * 8

        epsilon=1e-6

        #Define plane
        planeNormal = np.array([0, 0, 1])
        planePoint = np.array([0, 0, depth_est]) #Any point on the plane

        # convert to radians
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)

        # convert angles to vector components
        x = np.cos(yaw)*np.cos(pitch)
        y = np.sin(yaw)*np.cos(pitch)
        z = np.sin(pitch)

        #Define ray
        rayDirection = np.array([x, y, z])
        rayPoint = np.array([tx, ty, 0]) #Any point along the ray

        ndotu = planeNormal.dot(rayDirection)

        if abs(ndotu) < epsilon:
            print("no intersection or line is within plane")

        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * rayDirection + planePoint

        print("intersection at", Psi)


attn = Attention()

yaw = 2
pitch = -2
roll = -27
tx = 100
ty = 577
face_height = 89
face_width = 82
scale = 25  # pixels to feet
frame_height = 1080
frame_width = 1920
attn.predict_intersection(yaw, pitch, roll, tx, ty, face_height, face_width, scale)
