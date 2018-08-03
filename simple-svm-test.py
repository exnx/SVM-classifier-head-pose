from sklearn.svm import SVC
import sklearn
import numpy as np
import pandas as pd
import math
import csv
from sklearn.metrics import classification_report, confusion_matrix
import os

def create_test_houses_list(file_path):
    with open(file_path, "r") as f:
        hh_list = []
        for line in f:
            line = line.replace('/','-').strip()  # format the house hold list for matching
            hh_list.append(line)
    return hh_list

class SVM:

    def __init__(self, output_name):
        # if existing predictor file exists, remove it
        try:
            os.remove(output_name)
        except OSError:
            pass

        # write to csv (by appending to file)
        with open(output_name, 'a') as outfile:
            csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['subdir','frame_num','face_startX','face_startY','face_endX','face_endY',\
            'face_height','face_width','face_x_center','face_y_center','yaw','pitch','roll',\
            'raw_yaw','raw_pitch','raw_roll','predicted_label','attention_label',])

    def correct_data(self, frame_data):
        print('[Correcting data (removing improperly labeled samples)]...')

        corrected_data = []

        # iterate through each row in DF
        for index, row in frame_data.iterrows():
            attn_label = int(row[-1])  # isolate the attention label (last entry)

            if attn_label != -1:
                corrected_data.append(row)

        return pd.DataFrame(corrected_data)

    def read_csv(self, file_path):
        print('[Reading in CSV of entire dataset]...')
        frame_data = pd.read_csv(file_path, delim_whitespace=True)
        return frame_data

    # reserve test data from the entire data set, split training and validation data
    def reserve_test_data(self, frame_data, test_subdir):
        print('[Reserving test data from entire dataset]...')
        # track data with lists
        test_data_list = []
        train_and_validation_data_list = []

        # iterate through each row in DF
        for index, row in frame_data.iterrows():
            curr_subdir = row[0].strip()  # extract subdir (household and time) for testing

            # my test
            # 'HH0681-rgb_2017_3_18_22_0_0'
            # 'HH0647-rgb_2017_2_4_21_0_1'

            # isolate the household name only
            test_household = test_subdir[:6]
            curr_household = curr_subdir[:6]

            # only include the test_household matching hour too for test data
            # ie, dont train with other subdirs for different hours as well to avoid cross contamination
            if curr_subdir == test_subdir:
                test_data_list.append(row)

            # only data that is not the same house goes in the training
            elif curr_household != test_household:
                train_and_validation_data_list.append(row)

        # convert to DataFrame
        test_data = pd.DataFrame(data=test_data_list)
        train_and_validation_data = pd.DataFrame(data=train_and_validation_data_list)

        print('test_data shape', test_data.shape)
        print('train_and_validation_data shape', train_and_validation_data.shape)

        return (train_and_validation_data, test_data)

    def preprocess_test_data(self, test_data):

        # split feature and label
        X_test_raw, y_test_raw = self.split_x_and_y(test_data)

        # duplicate 3 columns to maintain actual yaw, pitch, roll
        X_test_raw['raw_yaw'] = X_test_raw['yaw']
        X_test_raw['raw_pitch'] = X_test_raw['pitch']
        X_test_raw['raw_roll'] = X_test_raw['roll']

        # normalize the X feature data, but keep the meta data
        X_train_norm = self.normalize_test_data(X_test_raw)

        return X_train_norm, y_test_raw

    def preprocess_training_data(self, training_data):
        print('[Preprocessing format for training]...')
        # need to remove all rows with 'NA'
        training_data = training_data.dropna()

        # split feature and label
        X_train_raw, y_train_raw = self.split_x_and_y(training_data)

        # remove the meta data from the feature vectors
        X_train_no_meta = self.drop_meta_data(X_train_raw)

        # normalize the data
        X_train_norm = self.normalize_train(X_train_no_meta)

        return (X_train_norm, y_train_raw)

    def train_svm(self, X_train, y_train):
        # data is now ready to train SVM
        print('[Training SVC ]...')
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train.values.ravel())  # use ravel to correct shape
        return svclassifier

    # need to take in raw data because we need to know which rows have 'NA'
    def predict_accuracy_on_data(self, svclassifier, X_test_processed, y_test_raw, output_name, test_hh):
        # must iterate one row at a time to set predictions for 'NA' rows (attn == 0)
        print('[Predicting accuracy]...')
        y_prediction = []
        csv_count = 0
        print('number of samples in this household', X_test_processed.shape[0])
        for index, row in X_test_processed.iterrows():

            y_pred_single = None
            y_test_single = int(y_test_raw.loc[index][0])  # grab the actual label

            # grab the relevant feature columns for predicting
            X_row = row[['face_height','face_width','face_x_center','face_y_center','yaw','pitch','roll']]
            new_num_cols = X_row.shape[0]

            # check if valid row
            if self.is_valid_row(X_row):
                X_row = np.array(X_row).reshape(-1,new_num_cols)  # reshape with updated num of cols
                y_pred_single = svclassifier.predict(X_row)[0]  # predict returns a list, so take first entry

            else:
                # print('non valid row found!')
                y_pred_single = 0

            y_prediction.append(y_pred_single)  # save the prediction for the row to running list

            # convert full row to list (with string data), appending predicted and actual labels
            predicted_full_entry = row.tolist()
            predicted_full_entry.append(y_pred_single)  # append the prediction
            predicted_full_entry.append(y_test_single)  # append the actual label, 1st entry, convert to int

            # write to csv with the feature vector (with string data) + the predicted label and the actual
            self.write_to_csv(predicted_full_entry, output_name)
            csv_count += 1

        print('csv_count for this test house:', csv_count)

        # convert all y_prediction values to a DF for the accuracy report
        y_prediction = pd.DataFrame(y_prediction)

        # check accuracy here
        print(y_prediction.shape)
        print(y_test_raw.shape)

        print('Accuracy results test hold out:', test_hh)
        print(classification_report(y_test_raw, y_prediction))
        print(confusion_matrix(y_test_raw, y_prediction))

### ----- helper functions

    def write_to_csv(self, row_with_predicted_label, output_name):
        # write to csv here
        with open(output_name, 'a') as outfile:
            csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row_with_predicted_label)

    def is_valid_row(self, row):

        for element in row:
            if not math.isnan(element):
                pass
            else:
                return False
        return True

    def split_x_and_y(self, training_data):
        X_train = training_data.drop(['attention_label'], axis=1)
        y_train = pd.DataFrame(training_data['attention_label'])
        return (X_train, y_train)

    def normalize_test_data(self, X_test):

        # ignore first 6 columns, and last 3 columns, these are meta deta
        for column in X_test.columns[6:-3]:

            # # had to append raw at the end, but need to ignore these when normalizing
            # if column != 'raw_yaw' or column != 'raw_pitch' or column != 'raw_roll':

            # calc mean and std
            std_dev = X_test[column].std()
            mean = X_test[column].mean()

            # scale each element in column
            X_test[column] = X_test[column].apply(lambda x: (x-mean)/std_dev)
        return X_test

    def normalize_train(self, X_data):
        # for each column in DF
        for column in X_data.columns:
            # calc mean and std
            std_dev = X_data[column].std()
            mean = X_data[column].mean()
            # scale each element in column
            X_data[column] = X_data[column].apply(lambda x: (x-mean)/std_dev)
        return X_data

    def drop_meta_data(self, X_data_raw):
        X_data = X_data_raw[['face_height','face_width','face_x_center','face_y_center','yaw','pitch','roll']]
        return X_data

# get test households list
test_hh_list = create_test_houses_list('formatted_households.txt')

# output the predicted labels for each sample
output_name = 'predicted_labels6.csv'
test = SVM(output_name)

# iterate through each house as the test holdout
for test_hh in test_hh_list:

    frame_data = test.read_csv('output_training_data_aug2.csv')  # data to train/test
    frame_data = test.correct_data(frame_data)  # remove the incorrect labeled entries

    # holdout test household, one at a time
    train_data, test_data = test.reserve_test_data(frame_data, test_hh)

    # preprocess training data for training svc
    X_train, y_train = test.preprocess_training_data(train_data)
    # train svm
    svc_classifier = test.train_svm(X_train, y_train)

    # preprocess test data, includes normalizing, and adding meta data for csv output
    X_test_processed, y_test_raw = test.preprocess_test_data(test_data)

    # test accuracy with a holdout for validation
    test.predict_accuracy_on_data(svc_classifier, X_test_processed, y_test_raw, output_name, test_hh)
