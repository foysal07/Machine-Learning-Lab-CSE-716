import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Read training and test dataset.
train = pd.read_csv('../Datasets/training-data-10-tuples.csv')
test = pd.read_csv('../Datasets/test-data-4-tuples.csv')

# LabelEncoder to convert categorical to numeric value.
number = LabelEncoder()

# Convert categorical values to numeric.
for i in train:
    train[i] = number.fit_transform(train[i].astype('str'))

# Split input and output columns; x = input columns, y = output columns.
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

# Do the same for test dataset.
for i in test:
    test[i] = number.fit_transform(test[i].astype('str'))

x_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Build and train SVM Classifier
SVM_Classifier = SVC(kernel='linear')
SVM_Classifier.fit(x_train, y_train)

# Predict on test-data
predicted = SVM_Classifier.predict(x_test)

# Print classification report
print(classification_report(y_test, predicted))

# Build confusion matrix
cfm = confusion_matrix(y_test, predicted)
# Calc accuracy
acc = accuracy_score(y_test, predicted)

# Print acc and cfm
print('Accuracy:', acc)
print('Prediction  no  yes')
print('        no  {}   {}'.format(cfm[0][0], cfm[0][1]))
print('       yes  {}   {}'.format(cfm[1][0], cfm[1][1]))