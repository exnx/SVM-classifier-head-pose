import os
import time
import re
import csv
import cv2
from imutils import face_utils
import numpy as np
import time
import math
import utils

class Annotation:

    def get_annotation_rows(self, annotations_path):

        annotation_rows = []

        with open(annotations_path, "rt") as file:
            # reader is all the rows loaded into memory at the beginning
            reader = csv.reader(file, delimiter=' ')
            counter = 0  # keep a counter of entries
            next(reader)  # skip the header

            # iterate through rows
            for row in reader:
                if row:  # only take care of empty rows
                    counter += 1  # increment counter

                    # strip white space from each entry
                    new_row = [entry.strip() for entry in row]

                    annotation_rows.append(new_row)

        return annotation_rows

class Display:

    def load_frame(self, frame_path):
        root = frame_path.split('/')[-3]  # just the root dir
        subdir = frame_path.split('/')[-2]  # just the household video dir
        frame_num = frame_path.split('/')[-1]  # just the frame file name

        # make sure that this directory is a subdir with HH prefix (for testing)
        # and check if file is a .jpg file
        if subdir[:2] == 'HH' and frame_num[-3:] == 'jpg':
            frame = cv2.imread(frame_path) # Read the image with OpenCV
            return frame

    def create_frames_set(self, frames_dir_path):
        print('[Creating frame set from path]...')
        exclude_prefixes = ('__', '.')  # exclusion prefixes
        frames_to_test = []  # will store all file names (and full path) here

        # walk through the directory and find all the files
        for root, dirs, files in os.walk(frames_dir_path):
            for name in files:
                if not name.startswith(exclude_prefixes):  # exclude hidden folders
                    # need to add append to list first
                    frames_to_test.append(os.path.join(root, name))

        # filter correct rows and put in a set instead
        frames_set = set()

        for file_path in frames_to_test:
            # need these for saving to csv
            root = file_path.split('/')[-3]  # just the root dir
            subdir = file_path.split('/')[-2]  # just the household video dir
            frame_num = file_path.split('/')[-1]  # just the frame file name

            # print(root, subdir, frame_num)

            # filtering out the right file types
            if subdir[:2] == 'HH' and frame_num[-3:] == 'jpg':
                # convert format
                frame_num = frame_num[:-4]
                frame_num = str(int(frame_num))
                relative_path = subdir + '/' + frame_num
                # print('relative path: ', relative_path)
                frames_set.add(relative_path)
                # print(relative_path)
        return frames_set

    def display_frames(self, frames_path, annotations_path):
        print('[Preparing data to display]...')
        exclude_prefixes = ('__', '.')  # exclusion prefixes

        # retrieve a set with all the relative frame paths in the directories to get features for testing/training
        frames_set = self.create_frames_set(frames_path)

        annotator = Annotation()
        annotation_rows = annotator.get_annotation_rows(annotations_path)

        # iterate through each annotation row
        for annotation_row_data in annotation_rows:

            subdir = annotation_row_data[0]
            frame_num = str(int(annotation_row_data[1]))
            annotation_row_name = subdir + '/' + frame_num

            if annotation_row_name in frames_set:

                frame_num_len = len(str(frame_num))

                # need to append .jpg to frame num, which has either 4 or 5 digits
                if frame_num_len < 5:  # less than 10k frames in subdir
                    formatted_frame_num = formatted_frame_num = str(frame_num).zfill(4) + '.jpg'
                elif frame_num_len > 4:  # might be 10k+ frames in a subdir
                    formatted_frame_num = str(frame_num) + '.jpg'

                file_path = '/'.join([frames_path, subdir, formatted_frame_num])

                # 0 - subdir
                # 1 - frame_num
                # 2 - face_startX
                # 3 - face_startY
                # 4 - face_endX
                # 5 - face_endY
                # 6 - face_height # normalized
                # 7 - face_width # normalized
                # 8 - face_x_center # normalized
                # 9 -face_y_center # normalized
                # 10 - yaw  # normalized
                # 11 - pitch # normalized
                # 12 - roll # normalized
                # 13 - raw_yaw
                # 14 raw_pitch
                # 15 - raw_roll
                # 16 - predicted_label
                # 17 - attention_label

                face_bounding_box = annotation_row_data[2:6]  # get face bb
                roll, pitch, yaw = annotation_row_data[13:16]  # get the head pose angles
                predicted_label = annotation_row_data[16]
                actual_label = annotation_row_data[17]

                # load the frame itself
                frame = self.load_frame(file_path)

                # if face box given, mark it up on image
                if face_bounding_box[0] != 'nan':

                    startX = int(float((face_bounding_box[0])))
                    startY = int(float((face_bounding_box[1])))
                    endX = int(float((face_bounding_box[2])))
                    endY = int(float((face_bounding_box[3])))

                    face_height = endY - startY

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)

                    if roll != 'nan':  # if there is a head pose

                        yaw_int = int(float(yaw))
                        pitch_int = int(float(pitch))
                        roll_int = int(float(roll))
                        # draw the axis
                        utils.draw_axis(frame, yaw_int, pitch_int, roll_int, tdx = (startX + endX)/2, tdy= (startY + endY)/2, size = face_height/2)

                        head_pose_text = 'yaw, pitch, roll:  ' + ', '.join([yaw, pitch, roll])
                        cv2.putText(frame, head_pose_text, (startX, startY-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        predicted_text = 'predicted: ' + str(predicted_label)
                        actual_text = 'actual: ' + str(actual_label)

                        # display the actual and predicted label on frame
                        cv2.putText(frame, predicted_text, (startX, startY-35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 2)
                        cv2.putText(frame, actual_text, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # if no face box, then display NA in left corner
                else:
                    cv2.putText(frame, 'No face box', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                # show the output image
                cv2.imshow("{}".format(file_path), frame)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()


def main(test_frames_path, annotations_path):
    displayer = Display()
    displayer.display_frames(test_frames_path, annotations_path)

if __name__ == "__main__":
    test_frames_path = 'test-frames'
    annotations_path = 'predicted_labels6.csv'
    frame_size = (1080,1920)
    main(test_frames_path, annotations_path)
