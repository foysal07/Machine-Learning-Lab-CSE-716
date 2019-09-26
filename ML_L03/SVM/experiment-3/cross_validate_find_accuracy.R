library(e1071)
library(caret)
library(RWeka)

data <- read.csv(file.choose(), header = TRUE)
kfolds <- createFolds(data$buysComputer, k=2)
sum = 0

for (i in kfolds) {
	train <- data[-i,]
	test <- data[i,]
	model <- J48(buysComputer~., data=train)
	prediction <- predict(model, test)
	cfMatrix <- confusionMatrix(data=prediction, test$buysComputer)
	sum <- sum + cfMatrix$overall[1]
}

accuracy <- sum / length(kfolds)
accuracy

