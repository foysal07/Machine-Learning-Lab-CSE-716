library(RWeka)
library(caret)
train <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/SVM/experiment-1/train.csv')
test <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/SVM/experiment-1/test.csv')
model <- train(buysComputer~., method='svmLinear', data = train)
prediction <- predict(model, test)
cfMatrix <- confusionMatrix(data=prediction, test$buysComputer)
cfMatrix