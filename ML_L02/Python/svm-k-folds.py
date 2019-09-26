from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC

# Read training and test dataset.
train = pd.read_csv('../Datasets/training-data-14-tuples.csv')

# LabelEncoder to convert categorical to numeric value.
number = LabelEncoder()

# Convert categorical values to numeric.
for i in train:
    train[i] = number.fit_transform(train[i].astype('str'))

# Create SVM Model
SVM_Classifier = SVC(kernel='linear')

# Create kFolds
kf = KFold(n_splits=2).split(train)

total = 0  # sum of the accuracies.
length = 0  # length of the kFolds

# Now loop for all the folds and predict, then sum the accuracies.
for train_indices, test_indices in kf:
    tmp_train = train.iloc[train_indices]
    tmp_test = train.iloc[test_indices]
    x_train = tmp_train.iloc[:, :-1]  # Upto last column exclusively.
    y_train = tmp_train.iloc[:, -1]  # Only the last column, i.e. buys_computer.
    x_test = tmp_test.iloc[:, :-1]
    y_test = tmp_test.iloc[:, -1]

    #  Train/Feed the dataset to the model.
    SVM_Classifier.fit(x_train, y_train)

    # Make prediction on the test set.
    predicted = SVM_Classifier.predict(x_test)
    print(predicted)
    print(y_train)
    print(y_test)
    acc = accuracy_score(y_test, predicted)
    # Sum the accuracy.
    total += acc
    # print(acc)
    # Keep track the length of the kFolds.
    length += 1

# Now take the average of the accuracies.
print('Accuracy:', total / length)
