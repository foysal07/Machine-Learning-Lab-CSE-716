library(RWeka)
library(caret)

data <- read.csv(file.choose())
test <- read.csv(file.choose())
classification <- J48(buysComputer~.,data=data)
pre1 <- predict(classification, test, type="class")
pre1

