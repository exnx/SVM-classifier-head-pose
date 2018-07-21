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
        self.frame_data = pd.read_csv('training_data_357pm.csv', delim_whitespace=True)
        self.face_box_index = 2
        self.pose_index = 6
        self.num_splits = 5

    # reserve test data from the entire data set, split training and validation data
    def reserve_test_data(self):
        print('[Reserving test data from entire dataset]...')
        # track data with lists
        test_data_list = []
        train_and_validation_data_list = []

        # iterate through each row in DF
        for index, row in self.frame_data.iterrows():
            curr_household = row[0]  # remove households for testing

            if curr_household == 'HH0647-rgb_2017_2_4_21_0_1' or curr_household == 'HH0490-rgb_2017_2_5_21_0_0':
                test_data_list.append(row)
            else:
                train_and_validation_data_list.append(row)

        # convert to DataFrame
        test_data = pd.DataFrame(data=test_data_list)
        train_and_validation_data = pd.DataFrame(data=train_and_validation_data_list)

        return (test_data, train_and_validation_data)

    def normalize(self, X_data):
        # for each column in DF
        for column in X_data.columns:
            # calc mean and std
            std_dev = X_data[column].std()
            mean = X_data[column].mean()
            # scale each element in column
            X_data[column] = X_data[column].apply(lambda x: (x-mean)/std_dev)
        return X_data

    # split the indices for training / validation
    def create_train_validation_splits(self, train_and_validation_data):
        print('[Creating split indices for the training and validation sets]...')
        # drop the attention fields (label)
        X_train_and_validation = train_and_validation_data.drop(['attention_label'], axis=1)

        # make sure to convert labels to data frame
        y_train_and_validation = pd.DataFrame(train_and_validation_data['attention_label'])

        # makes sure training and test samples separated by household (from train_and_validation_data)
        groups = train_and_validation_data['subdir']

        # GroupShuffleSplit to split by groups
        gss = GroupShuffleSplit(n_splits=self.num_splits, test_size=0.20)

        return (gss.split(X=X_train_and_validation, y=y_train_and_validation, groups=groups), X_train_and_validation, y_train_and_validation)

    # for each split, retrieve the data with the split indices
    # and remove strings and predicting attention for rows with NA vals
    def process_splits(self, train_validation_splits, X_train_and_validation, y_train_and_validation):
        fold_count = 1
        # for each split, prep the data for training
        for train_inds, validation_inds in train_validation_splits:
            print('[Starting processing on split: {}]...'.format(fold_count))

            # retrieve respective data from indices
            print('[Retrieving training data from indices]...')
            X_train_raw = X_train_and_validation.iloc[train_inds]
            y_train_raw = y_train_and_validation.iloc[train_inds]

            print('[Retrieving validation data from indices]...')
            X_validation_raw = X_train_and_validation.iloc[validation_inds]
            y_validation_raw = y_train_and_validation.iloc[validation_inds]

            print('[Dropping NA values in training data]...')
            # need to remove all the 'NA' values and string columns
            X_train, y_train = self.drop_na(X_train_raw, y_train_raw)
            print('[Dropping NA values in validation data]...')
            X_validation, y_validation = self.drop_na(X_validation_raw, y_validation_raw)

            # remove the meta data from the feature vectors
            print('[Removing meta data from X data]...')
            X_train = self.drop_meta_data(X_train)
            X_validation = self.drop_meta_data(X_validation)

            # normalize the data by fold, for now
            print('[Normalizing training and validation data]...')
            X_train = self.normalize(X_train)
            X_validation = self.normalize(X_validation)

            # data is now ready to train SVM
            print('[Starting SVC training for fold: {}]...'.format(fold_count))
            svclassifier = SVC(kernel='linear')
            svclassifier.fit(X_train, y_train.values.ravel())  # use ravel to correct shape

            # save the model to disk
            print('[Saving model for fold: {}]...'.format(fold_count))
            filename = 'fold_{}_model.sav'.format(fold_count)
            joblib.dump(svclassifier, filename)

            # pass the svclassifier instead of training data, along with validation data
            fold_accuracy = self.predict_accuracy_on_fold(fold_count, svclassifier, X_validation_raw, y_validation_raw)

            # increment fold counter
            fold_count += 1

    # need to take in raw data because we need to know which rows have 'NA'
    def predict_accuracy_on_fold(self, fold_count, svclassifier, X_validation_raw, y_validation_raw):
        # must iterate one row at a time to set predictions for 'NA' rows (attn == 0)
        print('[Predict_accuracy_on_fold number: {}]...'.format(fold_count))
        y_prediction= []
        for index, row in X_validation_raw.iterrows():
            y_pred_single = None
            row = row.drop(['frame_num']).drop(['subdir'])
            # check if valid row
            if self.is_valid_row(row):
                row = np.array(row).reshape(-1,7)
                y_pred_single = svclassifier.predict(row)[0]  # predict returns a list, so take first entry
                # print(y_pred_single)
            else:
                y_pred_single = 0
            y_prediction.append(y_pred_single)
        y_prediction = pd.DataFrame(y_prediction)
        # check accuracy here
        print(classification_report(y_validation_raw, y_prediction))
        # print(confusion_matrix(y_validation_raw, y_prediction))

    def is_valid_row(self, row):

        for element in row:
            if not math.isnan(element):
                pass
            else:
                return False
        return True

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

svm = SVM()
test_data, train_and_validation_data = svm.reserve_test_data()
splits, X_train_and_validation, y_train_and_validation = svm.create_train_validation_splits(train_and_validation_data)
svm.process_splits(splits, X_train_and_validation, y_train_and_validation)
