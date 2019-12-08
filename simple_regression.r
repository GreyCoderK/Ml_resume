#Data preprocessing in R

#Importing the dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,2:3]

#splitting the dataset into the training set and test set
#before this i need to install.packages('caTools')

#import caTools library
library(caTools)

#train the data set with 123 example
set.seed(999999999999999999999)

#set split ratio and the Y value
split = sample.split(dataset$Salary, SplitRatio = 2/3)

#split train and test
traing_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#mise a l'echelle
#don't include categorical dummy column
#if it's include it launch error

#traing_set[,2:3] = scale(traing_set[,2:3])
#test_set[,2:3] = scale(test_set[,2:3])

#regression linear
regressor = lm(formula = Salary ~ YearsExperience,
               data = traing_set)

#predis les données
y_pred = predict(regressor, newdata = test_set)

#visualisez les resultat des entrainnements
#install.packages('ggplot2')
library(ggplot2)

ggplot() +
  geom_point(aes(x = traing_set$YearsExperience, y = traing_set$Salary),
             colour = 'red') +
  geom_line(aes(x = traing_set$YearsExperience, y = predict(regressor, newdata = traing_set)),
             colour = 'blue') +
  ggtitle('Salaire vs Experience de travail(training_set)')+
  xlab("Année d'experience")+
  ylab('Salaire')


ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = traing_set$YearsExperience, y = predict(regressor, newdata = traing_set)),
            colour = 'blue') +
  ggtitle('Salaire vs Experience de travail(test_set)')+
  xlab("Année d'experience")+
  ylab('Salaire')
