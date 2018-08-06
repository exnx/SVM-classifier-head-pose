import cv2
import numpy as np


class Attention:

    def projection(self, yaw, pitch, roll, tx, ty, face_height, face_width, scale):

        depth_est = 0 # in ft
        pixels_per_foot = scale

        # assumed scale: 25 pixels to 1 ft
        # use width (in pixels) to estimate depth
        if face_width < 60:
            depth_est = pixels_per_foot * 18
        elif face_width < 100:
            depth_est = pixels_per_foot * 14
        elif face_width < 150:
            depth_est = pixels_per_foot * 10
        else:
            depth_est = pixels_per_footz * 8

        focal_length = 0.01204068  # 3.67mm in feet

        # intrinsics = np.vstack(((-focal_length, 0, 0), (0, -focal_length, 0), (0, 0, 1)))

        # get the 3d ray
        x = tx / focal_length
        y = ty / focal_length
        z = 1

        ray = np.array([x,y,z])
        ray_in_3d = ray*depth_est

        return (ray_in_3d)

attn = Attention()

# 918 802 8 -15 -6
# 662 62 63 1161 631 0 0 17 1
# 56 56 1137 646 -18 7 15
# 89 82 1357 577 2 -3 -27

yaw = 0
pitch = 0
roll = 0
tx = 100
ty = 577
face_height = 89
face_width = 82
scale = 1  # pixels to feet
frame_height = 1080
frame_width = 1920

projection_point = attn.projection(yaw, pitch, roll, tx, ty, face_height, face_width, scale)
print(projection_point)



# # predict distance away from center of tv, in feet
# x_feet = (x_pix - frame_width/2) / scale
# y_feet = (y_pix - frame_height/2) / scale
#
#
# # tv size: assume 50 in tv
# tv_height = 4
# tv_width = 3
#
# tolerance = 1.30  # 1.x
#
# if abs(x_feet) < tv_width/2 or abs(y_feet) < tv_height/2:
#     pred_attn = 2
# elif abs(x_feet) < tolerance * tv_width/2 or abs(y_feet) < tolerance * tv_height/2:
#     pred_attn = 1
# else:
#     pred_attn = 0
#
#
# print('2d intersection:' , x_feet, y_feet)
# print('attn prediction: ', pred_attn)
