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

        x_intersect = tx + depth_est * np.tan(np.radians(yaw))
        y_intersect = ty + depth_est * np.tan(np.radians(pitch))


        return (x_intersect, y_intersect)

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
scale = 25  # pixels to feet
frame_height = 1080
frame_width = 1920

x_pix, y_pix = attn.predict_intersection(yaw, pitch, roll, tx, ty, face_height, face_width, scale)

# predict distance away from center of tv, in feet
x_feet = (x_pix - frame_width/2) / scale
y_feet = (y_pix - frame_height/2) / scale


# tv size: assume 50 in tv
tv_height = 4
tv_width = 3

tolerance = 1.30  # 1.x

if abs(x_feet) < tv_width/2 or abs(y_feet) < tv_height/2:
    pred_attn = 2
elif abs(x_feet) < tolerance * tv_width/2 or abs(y_feet) < tolerance * tv_height/2:
    pred_attn = 1
else:
    pred_attn = 0


print('2d intersection:' , x_feet, y_feet)
print('attn prediction: ', pred_attn)
