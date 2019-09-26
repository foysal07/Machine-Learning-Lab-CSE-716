library(RWeka)
library(caret)
data <- read.csv('C:/Users/Md. Muhtadir Rahman/Desktop/SVM/experiment-3/dataset.csv')

kfolds <- createFolds(data$buys_computer, k = 2)
sum = 0
for(i in kfolds){
  trainData <- data[-i,]
  test <- data[i,]
  model <- train(buysComputer~., method='svmLinear', data = trainData)
  prediction <- predict(model, test)
  cfMatrix <- confusionMatrix(data = prediction, test$buysComputer)
  sum <- sum + cfMatrix$overall[1]
}
accuracy <- sum/length(kfolds)
accuracy
