import sklearn
import numpy as np
import pandas as pd
import csv
import random
from sklearn.externals import joblib # for saving model
import pickle
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class SVM:

    def __init__(self):
        print('[SVM object initialized]...')

        # reading in the feature extracted data with labels
        train_and_validation_data = pd.read_csv('sample.csv', delim_whitespace=True)
        self.face_box_index = 2
        self.pose_index = 6
        self.train_svm(train_and_validation_data)

    def normalize(self, X_data):
        print('[normalizing the data]...')
        print('X_data: ', X_data)
        print('X_data mean: ', X_data.mean(axis=0))
        print('X_data std: ', X_data.std(axis=0))

        print('X_data scaled ', preprocessing.scale(X_data, axis=0))

        # column_values = []
        #
        # for column in X_data.columns:
        #     print('column: ', column)
        #
        #     std_dev = X_data[column].std()
        #     mean = X_data[column].mean()
        #
        #     for element in X_data[column]:
        #
        #         entry = (element - mean)/std_dev
        #         print('entry: ', entry)
        #         column_values.append(entry)

                # print('element before: ', element)
                # print('mean column value: ', mean)
                # print('element after: ', element)

        # column_values = pd.DataFrame(column_values)
        # print(column_values)
        exit()

        return column_values

    # for each split, retrieve the data with the split indices
    # and remove strings and predicting attention for rows with NA vals
    def train_svm(self, train_and_validation_data):
        print('[Training SVM]...')

        # separate feature and labels
        X_train_raw = train_and_validation_data.drop(['attention_label'], axis=1)
        y_train_raw = pd.DataFrame(train_and_validation_data['attention_label'])

        # need to remove all the 'NA' values and string columns
        X_train, y_train = self.preprocess_for_training(X_train_raw, y_train_raw)

        # normalize the data
        X_train = self.normalize(X_train)

        # data is now ready to train SVM
        print('[Starting SVC training]...')
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train.values.ravel())  # use ravel to correct shape

    # need to take in raw data because we need to know which rows have 'NA'
    def predict_accuracy(self, fold_count, svclassifier, X_validation_raw, y_validation_raw):
        # must iterate one row at a time to set predictions for 'NA' rows (attn == 0)
        print('[Predict_accuracy_on_fold number: {}]...'.format(fold_count))
        y_prediction= []
        for index, row in X_validation_raw.iterrows():
            y_pred_single = None
            # check if valid row
            if self.is_valid_row(row):
                row = row.drop(['frame_num']).drop(['subdir'])
                row = np.array(row).reshape(-1,7)
                y_pred_single = svclassifier.predict(row)[0]  # predict returns a list, so take first entry
                # print(y_pred_single)
            else:
                y_pred_single = 0
            y_prediction.append(y_pred_single)
        y_prediction = pd.DataFrame(y_prediction)
        # check accuracy here
        # print(classification_report(y_validation_raw, y_prediction))
        # print(confusion_matrix(y_validation_raw, y_prediction))

    def is_valid_row(self, row):

        row = row.drop('subdir')
        row = row.drop('frame_num')

        for element in row:
            if not math.isnan(element):
                pass
            else:
                return False
        return True

    # takes in X and y, returns it without the NA fields and
    # removes meta data fields from X features
    def preprocess_for_training(self, X_data_raw, y_data_raw):
        X_data = []  # removed na rows
        y_data = []
        row_counter = 0  # track number of rows

        # iterate X_data_raw full row (since y_data doesn't have NA fields)
        # remove all the rows with 'NA'
        for _, row in X_data_raw.iterrows():  # note the _ gives wrong index, so use row_counter
            # check to see if the head pose angle was calculated or not
            if self.is_valid_row(row):
                # extract the relevant fields/label data (remove meta data)
                X_row = X_data_raw.iloc[row_counter].drop('subdir').drop('frame_num')
                y_row = y_data_raw.iloc[row_counter]

                # append to data sets
                X_data.append(X_row)
                y_data.append(y_row)

            # increment the row counter
            row_counter += 1

        # convert to dataframe
        X_data = pd.DataFrame(X_data)
        y_data = pd.DataFrame(y_data)

        return (X_data, y_data)

svm = SVM()
# test_data, train_and_validation_data = svm.reserve_test_data()
# make sure to normalize just on train and validation data only
# train_and_validation_data = svm.normalize(train_and_validation_data)
# splits, X_train_and_validation, y_train_and_validation = svm.create_train_validation_splits(train_and_validation_data)
# svm.process_splits(splits, X_train_and_validation, y_train_and_validation)
