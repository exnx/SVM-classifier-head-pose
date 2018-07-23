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

    # def __init__(self, output_name):
    #     # if existing predictor file exists, remove it
    #     try:
    #         os.remove(output_name)
    #     except OSError:
    #         pass

    def correct_data(self, frame_data):
        print('[Correcting data (removing improperly labeled samples)]...')

        corrected_data = []

        # iterate through each row in DF
        for index, row in frame_data.iterrows():
            # print(row.shape)
            attn_label = int(row[9])  # remove households for testing

            if attn_label != -1:
                corrected_data.append(row)

        return pd.DataFrame(corrected_data)

    def read_csv(self, file_path):
        print('[Reading in CSV of entire dataset]...')
        frame_data = pd.read_csv(file_path, delim_whitespace=True)
        return frame_data

    # reserve test data from the entire data set, split training and validation data
    def reserve_test_data(self, frame_data, test_household=None):
        print('[Reserving test data from entire dataset]...')
        # track data with lists
        test_data_list = []
        train_and_validation_data_list = []

        # iterate through each row in DF
        for index, row in frame_data.iterrows():
            curr_household = row[0].strip()  # remove households for testing

            # my test
            # 'HH0681-rgb_2017_3_18_22_0_0'
            # 'HH0647-rgb_2017_2_4_21_0_1'

            if curr_household == test_household:
                test_data_list.append(row)
            else:
                train_and_validation_data_list.append(row)

        # convert to DataFrame
        test_data = pd.DataFrame(data=test_data_list)
        # print(test_data.shape)
        train_and_validation_data = pd.DataFrame(data=train_and_validation_data_list)

        return (train_and_validation_data, test_data)

    def create_train_test_split(self, train_data, test_data):
        print('[Create train test split]...')
        X_train = train_data.drop(['attention_label'], axis=1)
        # make sure to convert labels to data frame
        y_train = pd.DataFrame(train_data['attention_label'])

        X_test = test_data.drop(['attention_label'], axis=1)
        # make sure to convert labels to data frame
        y_test = pd.DataFrame(test_data['attention_label'])

        return (X_train, y_train, X_test, y_test)

    def preprocess(self, X_train_raw, y_train_raw):
        print('[Preprocessing format for training]...')
        # need to remove all the 'NA' values and string columns
        X_train, y_train = self.drop_na(X_train_raw, y_train_raw)

        # remove the meta data from the feature vectors
        X_train = self.drop_meta_data(X_train)

        # normalize the data by fold, for now
        X_train = self.normalize(X_train)

        return (X_train, y_train)

    def train_svm(self, X_train, y_train):
        # data is now ready to train SVM
        print('[Training SVC ]...')
        svclassifier = SVC(kernel='linear')

        # X_train = X_train.drop(['face_height', 'face_width', 'face_x_center', 'face_y_center'], axis=1)

        svclassifier.fit(X_train, y_train.values.ravel())  # use ravel to correct shape
        return svclassifier

    # need to take in raw data because we need to know which rows have 'NA'
    def predict_accuracy_on_data(self, svclassifier, X_test, y_test, output_name, test_hh):
        print(X_test.shape)
        print(y_test.shape)

        # must iterate one row at a time to set predictions for 'NA' rows (attn == 0)
        print('[Predicting accuracy]...')
        y_prediction= []

        for index, row in X_test.iterrows():
            y_pred_single = None

            # drop the columns with strings
            X_row = row.drop(['subdir', 'frame_num'])

            # check if valid row
            if self.is_valid_row(X_row):
                X_row = np.array(X_row).reshape(-1,7)
                y_pred_single = svclassifier.predict(X_row)[0]  # predict returns a list, so take first entry
            else:
                # print('non valid row found!')
                y_pred_single = 0

            y_prediction.append(y_pred_single)  # save the prediction for the row to running list

            # convert full row to list (with string data)
            predicted_full_entry = row.tolist()
            predicted_full_entry.append(y_pred_single)  # append the prediction
            # write to csv with the feature vector (with string data) + the predicted label (not the actual)
            self.write_to_csv(predicted_full_entry, output_name)

        # convert all y_prediction values to a DF for the accuracy report
        y_prediction = pd.DataFrame(y_prediction)

        # check accuracy here
        print(y_prediction.shape)
        print(y_test.shape)

        print('Accuracy results test hold out:', test_hh)
        print(classification_report(y_test, y_prediction))
        print(confusion_matrix(y_test, y_prediction))

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

    def normalize_test_data(self, X_test):
        # X_test = X_test.drop(['subdir', 'frame_num'], axis=1)

        # X_test = X_test.drop(['face_height', 'face_width', 'face_x_center', 'face_y_center'], axis=1)

        # ignore the first 2 columns which are strings
        for column in X_test.columns[2:]:
            # calc mean and std
            std_dev = X_test[column].std()
            mean = X_test[column].mean()

            # scale each element in column
            X_test[column] = X_test[column].apply(lambda x: (x-mean)/std_dev)
        return X_test

    def normalize(self, X_data):

        # for each column in DF
        for column in X_data.columns:
            # calc mean and std
            std_dev = X_data[column].std()
            mean = X_data[column].mean()
            # scale each element in column
            X_data[column] = X_data[column].apply(lambda x: (x-mean)/std_dev)
        return X_data

    def drop_meta_data(self, X_data_raw):
        X_data = X_data_raw.drop(['subdir', 'frame_num'], axis=1)
        return X_data

    # takes in X and y, returns it without the NA fields and
    # removes meta data fields from X features
    def drop_na(self, X_data_raw, y_data_raw):
        # add the y column to the x data (to maintain NA rows for the y data)
        data_raw = X_data_raw.join(y_data_raw)

        # remove columns with NA
        data_raw = data_raw.dropna()
        X_data = data_raw.drop(['attention_label'], axis=1)
        y_data = pd.DataFrame(data_raw['attention_label'])

        return (X_data, y_data)


# get test households list
test_hh_list = create_test_houses_list('formatted_households.txt')

# testing calls
output_name = 'predicted_labels1.csv'

# iterate through each house as the test holdout
for test_hh in test_hh_list:
    # test = SVM(output_name)
    test = SVM()
    frame_data = test.read_csv('training_data_357pm.csv')
    frame_data = test.correct_data(frame_data)  # remove the incorrect labeled entries

    train_and_validation_data, test_data = test.reserve_test_data(frame_data, test_hh)
    # these are raw
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = test.create_train_test_split(train_and_validation_data, test_data)
    # preprocess training data for training svc
    X_train, y_train = test.preprocess(X_train_raw, y_train_raw)
    svc_classifier = test.train_svm(X_train, y_train)

    # normalize the test data before testing accuracy
    X_test_norm = test.normalize_test_data(X_test_raw)
    test.predict_accuracy_on_data(svc_classifier, X_test_norm, y_test_raw, output_name, test_hh)
