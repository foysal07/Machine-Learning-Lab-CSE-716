library(RWeka)
library(caret)
data <- read.csv('C:/Users/Md. Muhtadir Rahman/Desktop/SVM/experiment-2/train.csv')
test <- read.csv('C:/Users/Md. Muhtadir Rahman/Desktop/SVM/experiment-2/test.csv')

classification <- train(buysComputer~., method="svmLinear", data = data)
prediction <- predict(classification, test)
prediction
