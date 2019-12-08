#Data preprocessing in R

#Importing the dataset
dataset = read.csv('Data.csv')

#Importing the dataset
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'),
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'),
                         labels = c(0,1))

#splitting the dataset into the training set and test set
#before this i need to install.packages('caTools')

#import caTools library
library(caTools)

#train the data set with 123 example
set.seed(123)

#set split ratio and the Y value
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

#split train and test
traing_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#mise a l'echelle
#don't include categorical dummy column
#if it's include it launch error
traing_set[,2:3] = scale(traing_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])
