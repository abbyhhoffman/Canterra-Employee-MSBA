#GOAL:You are required to model the probability of attrition and draw insights from your analysis. 
library(readxl)
library(tidyverse)
library(ggplot2)
library(magrittr)
library(broom)
library(corrplot)
library(stargazer)
library(moments)
library(rsample)  #for resampling procedures
library(caret)    #for resampling and model training
library(h2o)      #for resampling and model traininginst
library(performance)
library(dplyr)
library(class)
library(caret)
library(data.table)
library(ROCR)
library(ROSE) #Sampling-over and under, ROC and AUC curve

#read data 

# load data
employee_data <- read_excel("Employee_Data_Project (1).xlsx")


#Data Wrangling 

# edit data types
employee_data <- employee_data %>%
  mutate(TotalWorkingYears = as.numeric(TotalWorkingYears),
         NumCompaniesWorked = as.numeric(NumCompaniesWorked),
         Education = as.numeric(Education))

employee_data <- employee_data %>%
  mutate(Attrition = as.factor(Attrition),
         BusinessTravel = as.factor(BusinessTravel),
         Gender = as.factor(Gender),
         JobLevel = as.factor(JobLevel),
         MaritalStatus = as.factor(MaritalStatus),
         EnvironmentSatisfaction = as.numeric(EnvironmentSatisfaction),
         JobSatisfaction = as.numeric(JobSatisfaction))

# convert dependent variable to binary instead of yes/no (for logistic model purposes)
employee_data <- employee_data %>%
  mutate(Attrition = case_when(
    Attrition == "Yes" ~ 1,
    Attrition == "No" ~ 0))

# reorder data
col_order <- c("Age", "BusinessTravel", "DistanceFromHome", "Education", "EmployeeID", "Gender", "JobLevel", "MaritalStatus", "Income", "NumCompaniesWorked", "StandardHours", "TotalWorkingYears", "TrainingTimesLastYear", "YearsAtCompany", "YearsWithCurrManager", "EnvironmentSatisfaction", "JobSatisfaction", "Attrition")
employee_data <- employee_data[, col_order]




sapply(employee_data,function(x) sum(is.na(x)))


#Replace Numeric Values with Medians
employee_data$TotalWorkingYears[is.na(employee_data$TotalWorkingYears)] = median(employee_data$TotalWorkingYears, na.rm = TRUE)
employee_data$NumCompaniesWorked[is.na(employee_data$NumCompaniesWorked)] = median(employee_data$NumCompaniesWorked, na.rm = TRUE) 
#Replace Numeric Values with Mean
employee_data$JobSatisfaction[is.na(employee_data$JobSatisfaction)] = mean(employee_data$JobSatisfaction, na.rm = TRUE)
employee_data$EnvironmentSatisfaction[is.na(employee_data$EnvironmentSatisfaction)] = mean(employee_data$EnvironmentSatisfaction, na.rm = TRUE) 


sapply(employee_data,function(x) sum(is.na(x))) # NO missing values! 

# train test split (stratified random sampling)
set.seed(1000)
# set the index using the caret splitting
index <- createDataPartition(y = employee_data$Attrition,
                             p = 0.70,
                             list = FALSE)

# training set
employee_data_train <- employee_data[index, ]

# testing set
employee_data_test <- employee_data[-index, ]


#Fairly Equal Distributions of Attrition for Test & Train # No fixing required
table(employee_data_train$Attrition)
table(employee_data_test$Attrition)
prop.table(table(employee_data_train$Attrition))
prop.table(table(employee_data_test$Attrition))


############ Over-sampling, under-sampling, combination of over- and under-sampling.#####################
data_under <- ovun.sample(Attrition ~ ., data = employee_data_train, method = "under", p=0.5, seed=1000)$data
#
data_over <- ovun.sample(Attrition ~ ., data = employee_data_train, method = "over", p=0.5, seed=1000)$data
#
data_ou <- ovun.sample(Attrition ~ ., data = employee_data_train, method = "both",N = nrow(employee_data_train), p=0.5, seed=1000)$data
#

# ROSE METHOD # 

data_rose <- ROSE(formula = Attrition ~.,
                                 data = employee_data_train,
                                 p = 0.5,
                                 seed = 1000)$data

# explore distribution of the dependent variable
employee_data_train %>%
  group_by(Attrition) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count, na.rm = TRUE))

data_under %>%
  group_by(Attrition) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count, na.rm = TRUE))

data_over %>%
  group_by(Attrition) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count, na.rm = TRUE))

data_ou %>%
  group_by(Attrition) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count, na.rm = TRUE))

data_rose %>%
  group_by(Attrition) %>%
  summarize(count = n()) %>%
  mutate(proportion = count / sum(count, na.rm = TRUE))

# choosing the best method for weighting samples
model_all <- glm(Attrition ~ ., family = "binomial", employee_data_train)
model_all_under <- glm(Attrition ~ ., family = "binomial", data_under)
model_all_over <- glm(Attrition ~ ., family = "binomial", data_over)
model_all_overunder <- glm(Attrition ~ ., family = "binomial", data_ou)
model_all_rose <- glm(Attrition ~ ., family = "binomial", data_rose)

summary(model_all) #AIC 2333.3
summary(model_all_under) #AIC 1183.3 #WINNERWINNER LOL
summary(model_all_over) #AIC 5902.2
summary(model_all_overunder) #AIC 3575.3
summary(model_all_rose) # AIC 3728.3

######## UNDER WEIGHT OF ATTRITION MODEL IS BEST #######


# model 0: management's hypothesis
model_0 <- glm(Attrition ~ JobSatisfaction + TotalWorkingYears + YearsAtCompany, family = "binomial", data_under)

# make predictions
model_0_train_preds <-predict(model_0, newdata = employee_data_train, type = "response")
employee_data_train_model_0 <- employee_data_train
employee_data_train_model_0$preds <- ifelse(model_0_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_model_0$preds),
                reference = as.factor(employee_data_train_model_0$Attrition),
                positive = "1")


#Area under the curve (AUC): 0.656
# VIF
vif(model_0)

# summary
summary(model_0)



#AIC 1300
#ACCURACY 62.81%
#Sensitivity 69.65%
#Balanced Accuracy 65.59%

# 95% CI : (0.6108, 0.6452)
#No MultiCollinearity




# model 1: employee demographics
model_1 <- glm(Attrition ~ Gender + Education + Age, family = "binomial", data_under)

# make predictions
model_1_train_preds <-predict(model_1, newdata = employee_data_train, type = "response")
employee_data_train_model_1 <- employee_data_train
employee_data_train_model_1$preds <- ifelse(model_1_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_model_1$preds),
                reference = as.factor(employee_data_train_model_1$Attrition),
                positive = "1")

#Area under the curve (AUC): 0.606
# VIF
vif(model_1)

# summary
summary(model_1)


#AIC 1332.6
#ACCURACY 58.02%
#Sensitivity 64.36%
#Balanced Accuracy 60.59%

#    95% CI : (0.5625, 0.5977)
#No MultiCollinearity

#AGE IS ONLY SIGNIFICANT FACTOR OF THREE TESTED



# model 2: management's hypothesis + employee demographics
model_2 <- glm(Attrition ~ JobSatisfaction + TotalWorkingYears + YearsAtCompany + Gender + Education + Age, family = "binomial", data_under)

# make predictions
model_2_train_preds <-predict(model_2, newdata = employee_data_train, type = "response")
employee_data_train_model_2 <- employee_data_train
employee_data_train_model_2$preds <- ifelse(model_2_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_model_2$preds),
                reference = as.factor(employee_data_train_model_2$Attrition),
                positive = "1")



# VIF
vif(model_2)

# summary
summary(model_2)


#AIC 1299.3
#ACCURACY 61.84%
#Sensitivity 67.01%
#Balanced Accuracy 63.93%

# 95% CI : (0.601, 0.6356)
#No MultiCollinearity

#Job Satisaction Total Working Years and Age are signifcant at varying levels



# model 3 My hypothesis
model_3 <- glm(Attrition ~ JobSatisfaction + TotalWorkingYears + YearsAtCompany + EnvironmentSatisfaction + TrainingTimesLastYear, family = "binomial", data_under)

# make predictions
model_3_train_preds <-predict(model_3, newdata = employee_data_train, type = "response")
employee_data_train_model_3 <- employee_data_train
employee_data_train_model_3$preds <- ifelse(model_3_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_model_3$preds),
                reference = as.factor(employee_data_train_model_3$Attrition),
                positive = "1")


#Area under the curve (AUC): 0.648
# VIF
vif(model_3)

# summary
summary(model_3)

#AIC 1291
#ACCURACY 63.52%
#Sensitivity 66.60%
#Balanced Accuracy 64.77%

#  95% CI : (0.618, 0.6523)
#No MultiCollinearity



## Hypothesis variables + My hypothesis
model_4 <- glm(Attrition ~ JobSatisfaction + TotalWorkingYears + YearsAtCompany + Gender + Education + Age+ EnvironmentSatisfaction + TrainingTimesLastYear, family = "binomial", data_under)
summary(model_4)

# make predictions
model_4_train_preds <-predict(model_4, newdata = employee_data_train, type = "response")
employee_data_train_model_4 <- employee_data_train
employee_data_train_model_4$preds <- ifelse(model_4_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_model_4$preds),
                reference = as.factor(employee_data_train_model_4$Attrition),
                positive = "1")



# VIF
vif(model_4)

# summary
summary(model_4)

#Area under the curve (AUC): 0.662

#AIC 1289.9
#ACCURACY 64.76%
#Sensitivity 68.43%
#Balanced Accuracy 66.25%

# 95% CI : (0.6304, 0.6644)
#No MultiCollinearity

# STARGAZER

stargazer(model_0, model_1, model_2,model_3,model_4, align=T, type="text", no.space=TRUE)
stargazer(model_0, model_1, model_2, model_3, model_4, align=T, type="latex", no.space=TRUE)

#################################################################################################




stepmodel = step(model_4, direction="both")




# BEST MODEL 
#AIC=1285.49
#Attrition ~ JobSatisfaction + TotalWorkingYears + Age + EnvironmentSatisfaction + TrainingTimesLastYear


summary(stepmodel)


## STEP MODEL -> FINAL MODEL 
final_model <- glm(formula = Attrition ~ JobSatisfaction + TotalWorkingYears + 
                     Age + EnvironmentSatisfaction + TrainingTimesLastYear, family = "binomial", 
                   data = data_under)


summary(final_model) #AIC: 1285.5

vif(final_model) #ALL UNDER 2


# make predictions: train
final_model_train_preds <-predict(final_model, newdata = employee_data_train, type = "response")
employee_data_train_final_model <- employee_data_train
employee_data_train_final_model$preds <- ifelse(final_model_train_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_train_final_model$preds),
                reference = as.factor(employee_data_train_final_model$Attrition),
                positive = "1")

# ROC curve
roc.curve(employee_data_train_final_model$Attrition,
          final_model_train_preds, add.roc=FALSE, col=1)
#Area under the curve (AUC): 0.698

#AIC 1294.2
#ACCURACY 64.44%
#Sensitivity 68.02%
#Balanced Accuracy 65.87%

# 95% CI : (0.6268, 0.6609)
#No MultiCollinearity


###################### TEST DATA #################################
final_model_test_preds <-predict(final_model, newdata = employee_data_test, type = "response")
employee_data_test$preds <- ifelse(final_model_test_preds >= 0.5, 1, 0)

# confusion matrix
confusionMatrix(data = as.factor(employee_data_test$preds),
                reference = as.factor(employee_data_test$Attrition),
                positive = "1")

# ROC curve
roc.curve(employee_data_test$Attrition,
          final_model_test_preds, add.roc=T, col=2,lwd=2, lty=2)
legend("topleft", c("Training Set", "Test Set"), 
                                                              col=1:2, lty=1:2, lwd=2)


#Area under the curve (AUC): 0.672

#ACCURACY 64.32%
#Sensitivity 62.27%
#Balanced Accuracy 63.50%

#95% CI : (0.6167, 0.6691)
#No MultiCollinearity


