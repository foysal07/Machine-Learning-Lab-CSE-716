library(RWeka)
library(caret)

#this program uses RWeka and caret
#Holdout method

train <- read.csv(file.choose(), header = TRUE)
test <- read.csv(file.choose(), header = TRUE)
model <- J48(class~., data=train)
prediction <- predict(model, test)
cfMatrix <- confusionMatrix(data = prediction, test$class)
cfMatrix

