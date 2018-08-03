from sklearn.metrics import classification_report, confusion_matrix
import csv

# read in prediction labels

def read_labels(labels_path):

    predictions = []
    actuals = []
    entry_set = set()
    first_line = True

    with open(labels_path, "r") as f:
        for row in f:
            if first_line:
                first_line = False
                continue

            row = list(row.split())
            predictions.append(row[-2])
            actuals.append(row[-1])

    return (predictions, actuals)

def class_report(predictions, actuals):

    print(classification_report(predictions, actuals))
    print(confusion_matrix(predictions, actuals))

predictions, actuals = read_labels('predicted_labels5.csv')
class_report(predictions, actuals)
