library(RWeka)
library(nnet)
 
data <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/ANN/experiment-2/train.csv')
test <- read.csv('C:/Users/FOYSAL/Desktop/16701007-Machine Learing Lab/ANN/experiment-2/test.csv')
classification <- nnet(buysComputer~., size=2, data = data)
prediction <- predict(classification, test, type = "class")
prediction