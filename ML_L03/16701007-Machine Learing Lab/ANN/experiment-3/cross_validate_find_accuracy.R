library(RWeka)
library(caret)
library(nnet)
 
data <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/ANN/experiment-3/dataset.csv')
kfolds <- createFolds(data$buysComputer, k = 2)
sum = 0
for(i in kfolds){
  train <- data[-i,]
  test <- data[i,]
  model <- train(buysComputer~., method='nnet', data = train)
  prediction <- predict(model, test)
  cfMatrix <- confusionMatrix(data = prediction, test$buysComputer)
  sum <- sum + cfMatrix$overall[1]
}
accuracy <- sum/length(kfolds)
accuracy
