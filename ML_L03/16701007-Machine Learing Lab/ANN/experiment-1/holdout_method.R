library(RWeka)
library(caret)
library(nnet)
 
train <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/ANN/experiment-1/train.csv')
test <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/ANN/experiment-1/test.csv')
model <- train(class~., method='nnet', data = train)
prediction <- predict(model, test)
cfMatrix <- confusionMatrix(data=prediction, test$class)
cfMatrix
