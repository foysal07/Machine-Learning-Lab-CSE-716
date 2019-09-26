import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Read training and test dataset.
train = pd.read_csv('../Datasets/training-data-14-tuples.csv')
test = pd.read_csv('../Datasets/unknown-classed-tuple.csv')

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

x_test = test.iloc[:]

# Build and train SVM Classifier
SVM_Classifier = SVC(kernel='linear')
SVM_Classifier.fit(x_train, y_train)

# Do a prediction on unknown dataset.
predictions = SVM_Classifier.predict(x_test)

# Print the predicted results.
for i in predictions:
    print('Prediction: yes') if i == 1 else print('Prediction: no')

