import os
import time
import re
import csv
import cv2
from imutils import face_utils
import numpy as np
import time
import math

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

    def get_all_predicted_labels(self, predicted_labels_path):

        pred_label_rows = []

        with open(predicted_labels_path, "rt") as file:
            # reader is all the rows loaded into memory at the beginning
            reader = csv.reader(file, delimiter=' ')

            # iterate through rows
            for row in reader:
                if row:  # only take care of empty rows
                    # strip white space from each entry
                    new_row = [entry.strip() for entry in row]

                    pred_label_rows.append(new_row)
        return pred_label_rows

    def get_single_predicted_label(self, predicted_label_rows, subdir, frame_num):
        label_matches = []  # return this later
        match_found = False

        # iterate through all rows
        for row in predicted_label_rows:
            if len(row) > 2 and row[0] == subdir and row[1] == frame_num:
                label_matches.append(row[9])
                match_found = True
                print('match is found!')

        return match_found, label_matches  # return the matches

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

    def display_frames(self, frames_path, annotations_path, predicted_labels_path):
        print('[Preparing data to display]...')
        exclude_prefixes = ('__', '.')  # exclusion prefixes

        # retrieve a set with all the relative frame paths in the directories to get features for testing/training
        frames_set = self.create_frames_set(frames_path)

        annotator = Annotation()
        annotation_rows = annotator.get_annotation_rows(annotations_path)
        predicted_labels_rows = annotator.get_all_predicted_labels(predicted_labels_path)

        # iterate through each annotation row
        for annotation_row_data in annotation_rows:

            subdir = annotation_row_data[0]
            frame_num = str(int(annotation_row_data[1]))
            annotation_row_name = subdir + '/' + frame_num

            if annotation_row_name in frames_set:
                # create path with full format to find frame
                formatted_frame_num = str(frame_num).zfill(4) + '.jpg'
                file_path = '/'.join([frames_path, subdir, formatted_frame_num])

                face_bounding_box = annotation_row_data[2:6]  # get face bb
                roll, pitch, yaw = annotation_row_data[6:9]  # get the head pose angles
                attention_label = 'None'

                # load the frame itself
                frame = self.load_frame(file_path)

                # if face box given, mark it up on image
                if face_bounding_box[0] != 'NA':
                    startX = int(face_bounding_box[0])
                    startY = int(face_bounding_box[1])
                    endX = int(face_bounding_box[2])
                    endY = int(face_bounding_box[3])

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    head_pose_text = ', '.join([roll, pitch, yaw])
                    cv2.putText(frame, head_pose_text, (startX, startY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                    # get the predicted label
                    label_match, labels_in_frame = annotator.get_single_predicted_label(predicted_labels_rows, subdir, frame_num)
                    # draw the predicted attention label
                    if label_match:  # if labels found, display each one at the same time
                        all_labels_text = ''
                        for label in labels_in_frame: # loop through all labels
                            all_labels_text = all_labels_text + label + ' '  # create text of all labels
                    else:
                        all_labels_text = 'no label found'
                    # display the label on frame
                    cv2.putText(frame, str(all_labels_text), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                # if no face box, then display NA in left corner
                else:
                    cv2.putText(frame, 'No face box', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                # show the output image
                cv2.imshow("{}".format(file_path), frame)
                key = cv2.waitKey(0) & 0xFF


def main(test_frames_path, annotations_path, predicted_labels_path):
    displayer = Display()
    displayer.display_frames(test_frames_path, annotations_path, predicted_labels_path)

if __name__ == "__main__":
    test_frames_path = 'test-frames'
    annotations_path = 'annotation_sub_label.csv'
    predicted_labels_path = 'predicted_labels.csv'
    frame_size = (1080,1920)
    main(test_frames_path, annotations_path, predicted_labels_path)
