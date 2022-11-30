#######EXAMPLE 1. ICECREAM SALES DATA#########

icecream <- read.csv("C:/Users/Sudipta Dasmohapatra/Documents/Georgetown/Courses/MSBA/OPIM 602/Datasets/IceCream_sales.csv", sep = ",", stringsAsFactors = FALSE)

head(icecream, n=10)

model1 <- glm(buy_ind ~ Age, family = "binomial", data = icecream)  

summary(model1)  

#test for deviance

anova(model1, test="Chisq")

exp(coef(model1))
predict(model1, data.frame(Age = c(12)), type = "response")

#The 95% confidence limits indicate that you are 95% confident that the true logodds is between a certain range
#Because the 95% confidence interval does not include 1 so the odds ratio is significant at 0.05 alpha level. 

exp(confint(model1))

#Predicting on training data itself
buy_pred <- predict(object = model1, newdata = icecream, type = "response")
head(buy_pred)
summary(buy_pred)

#Lets set the threshold to 0.50 for predicting into 1
icecream$predicted<-ifelse(buy_pred>=0.5, 1, 0)
head(icecream$predicted)

#Accuracy
table(icecream$buy_ind, icecream$predicted)
prop.table(table(icecream$buy_ind, icecream$predicted))

confusionMatrix(data = as.factor(icecream$predicted),
                reference =  as.factor(icecream$buy_ind),
                positive = "1")

#ROC Curve
icecream_roc <- prediction(predictions = buy_pred, labels = icecream$buy_ind)
icecream_rocperf <- performance(icecream_roc , "tpr" , "fpr")
plot(icecream_rocperf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

#ROC Curve#
library(ROSE)
roc.curve(icecream$buy_ind, buy_pred)

#######EXAMPLE 2. LOAN PREDICTION DATA#########

#Default is equal to 1 and not default = 0 in the dataset in Class variable
#Importing the dataset

loan<- read.csv("C:/Users/Sudipta Dasmohapatra/Documents/Georgetown/Courses/MSBA/OPIM 602/Week4/Loan Default Prediction2.csv", 
                sep = ",", stringsAsFactors = FALSE)

dim(loan)
head(loan)

table(loan$SeriousDlqin2yrs)
prop.table(table(loan$SeriousDlqin2yrs))
#Imbalanced dataset with only 6.7% of loan default

summary(loan)

#1. Data splitting into training and test (70:30)
#Create split, any column is fine
library(caret)
set.seed(123)  # for reproducibility
index <- createDataPartition(loan$SeriousDlqin2yrs, p = 0.7, 
                             list = FALSE)
train <- loan[index, ]
test  <- loan[-index, ]
table(train$SeriousDlqin2yrs)
table(test$SeriousDlqin2yrs)
prop.table(table(train$SeriousDlqin2yrs))
prop.table(table(test$SeriousDlqin2yrs))

#2. Missing values
sapply(train,function(x) sum(is.na(x)))
sapply(test,function(x) sum(is.na(x)))

#Replacing num_dependents by median
train$Num_dependents[is.na(train$Num_dependents)] = median(train$Num_dependents, na.rm = TRUE)
test$Num_dependents[is.na(test$Num_dependents)] = median(test$Num_dependents, na.rm = TRUE)

#Replacing MonthlyIncome with median (<20% of data)
train$MonthlyIncome[is.na(train$MonthlyIncome)] = median(train$MonthlyIncome, na.rm = TRUE)
test$MonthlyIncome[is.na(test$MonthlyIncome)] = median(test$MonthlyIncome, na.rm = TRUE)

#Check again to make sure all replacements are done
sapply(train,function(x) sum(is.na(x)))
sapply(test,function(x) sum(is.na(x)))

#3. DOWNSAMPLING
install.packages("ROSE")
library(ROSE)
data_balanced_under <- ovun.sample(SeriousDlqin2yrs ~ ., data = train, method = "under",N = 15000)$data
table(data_balanced_under$SeriousDlqin2yrs)

#Downsampling downsamples the 0 category

#4. LOGISTIC MODEL#
Logistic_model <-glm(SeriousDlqin2yrs~ Utilization+Age+Num_loans+Num_dependents+MonthlyIncome+Num_Savings_Accts+DebtRatio, 
                     data=data_balanced_under, family=binomial())
summary(Logistic_model)

#Check multicollinearity
library(car)
vif(Logistic_model)

#As you can see, all the values of VIF for all the variables are less than 5, we need not reject any variable
#utilization, number of loans and debtratio are not significant

#Using Stepwise regression

stepmodel = step(Logistic_model, direction="both")

formula(stepmodel)
summary(stepmodel)

#Odds ratio
exp(coef(stepmodel))

#5. Prediction on the test set

loan_pred <- predict(object = stepmodel, newdata = test, type = "response")
head(loan_pred)
summary(loan_pred)

#Lets set the threshold to 0.50 for predicting into 1
test$predicted<-ifelse(loan_pred>=0.5, 1, 0)
head(test$predicted)

#6. Model Performance
#Accuracy
table(test$SeriousDlqin2yrs, test$predicted)

confusionMatrix(data = as.factor(test$predicted),
                reference =  as.factor(test$SeriousDlqin2yrs),
                positive = "1")

#As it can be seen , the sensitivity or the power of the model to detect success cases is average (54%)
#when the threshold is 0.5. But what about other thresholds?

#ROC
roc_pred <- prediction(predictions = loan_pred  , labels = test$SeriousDlqin2yrs)
roc_perf <- performance(roc_pred , "tpr" , "fpr")
plot(roc_perf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

# AUC (TWO WAYS)
roc.curve(test$SeriousDlqin2yrs, loan_pred)
as.numeric(performance(roc_pred, "auc")@y.values)

## Area under the curve (AUC): 0.641

#So by reducing the threshold, we increase the number of “positive” cases, 
#and thus increase sensitivity and decrease specificity. Maybe 0.4 is a good trade-off for our threshold.

test$predicted2<-ifelse(loan_pred>=0.45, 1, 0)
head(test$predicted2)

confusionMatrix(data = as.factor(test$predicted2),
                reference =  as.factor(test$SeriousDlqin2yrs),
                positive = "1")

#The sensitivity is now around 67% but the specificity is lowered around 53%. 
#The overal accuracy is 54%, so the missclassification is around 48%.

#ROSE (Random Over-Sampling Examples) aids the task of binary classification in the presence of rare classes. 
#It produces a synthetic, possibly balanced, sample of data simulated according to a smoothed-bootstrap approach.


###### You can also generate new balanced data by ROSE
library(ROSE)
data_rose <- ROSE(SeriousDlqin2yrs ~ ., data = train)$data


# check (im)balance of new data
table(data_rose$SeriousDlqin2yrs)

# train logistic regression on balanced data
rose_model <- glm(SeriousDlqin2yrs ~ ., data=data_rose, family="binomial")
# use the trained model to predict test data
rose_pred <- predict(rose_model, newdata=test,
                     type="response")
#Model Performance
test$predicted3<-ifelse(rose_pred>=0.5, 1, 0)

confusionMatrix(data = as.factor(test$predicted3),
                reference =  as.factor(test$SeriousDlqin2yrs),
                positive = "1")
roc_pred <- prediction(predictions = rose_pred  , labels = test$SeriousDlqin2yrs)
roc_perf <- performance(roc_pred , "tpr" , "fpr")

plot(roc_perf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

roc.curve(test$SeriousDlqin2yrs, rose_pred, add.roc=TRUE, col="red")
#639

#According to AUC values, the ROSE model is not better than the downsampling model. 


###################TITANIC DATASET#################

titanic<- read.csv("C:/Users/Sudipta Dasmohapatra/Documents/Georgetown/Courses/MSBA/OPIM 602/Datasets/titanic.csv", sep = ",", stringsAsFactors = TRUE)

dim(titanic)
head(titanic)

table(titanic$Survived)
prop.table(table(titanic$Survived))

summary(titanic)

#1. Data splitting into training and test (70:30)
#Create split, any column is fine
set.seed(123)  # for reproducibility
index <- createDataPartition(titanic$Survived, p = 0.7, 
                             list = FALSE)
train <- titanic[index, ]
test  <- titanic[-index, ]
table(train$Survived)
table(test$Survived)
prop.table(table(train$Survived))
prop.table(table(test$Survived))

#In the train dataset, the proportion of our data was around 61.61% not survived and around 38.38% survived. 
#It's balanced enough.

#2. Check the if there is missing value in data and the data type of data.

colSums(is.na(train))

#There is 125 missing value in age column. 
#It was around 20% of our data. Instead of removing the data. 
#let's try to replace the age number with the mean of age.

train_clean <- train %>%
  mutate(Age = if_else(is.na(Age), mean(Age, na.rm = TRUE), Age))
colSums(is.na(train_clean))

test_clean <- test %>%
  mutate(Age = if_else(is.na(Age), mean(Age, na.rm = TRUE), Age))
colSums(is.na(test_clean))

#In this case, Survived is the target variable
#Let's select some predictor variable and change the data type.
library(dplyr)
data_train <- train_clean %>% 
  select(-c(PassengerId, Name, Ticket, Cabin)) %>% 
  mutate(Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         SibSp = as.integer(SibSp),
         Parch = as.integer(Parch),
         Embarked = as.factor(Embarked))

data_test <- test_clean %>% 
  select(-c(PassengerId, Name, Ticket, Cabin)) %>% 
  mutate(Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         SibSp = as.integer(SibSp),
         Parch = as.integer(Parch),
         Embarked = as.factor(Embarked))

#Create dummies
library(dummies)
train_data <- dummy.data.frame(data_train, names=c("Pclass", "Sex","Embarked"), sep="_")
test_data <-dummy.data.frame(data_test, names=c("Pclass", "Sex","Embarked"), sep="_")
str(train_data)
str(test_data)

#Let's check the structure of data train by using str()
str(data_train)

#3. Logistic Regression Model
m_LR<- glm(Survived~., train_data, family = "binomial") 
summary(m_LR)

#From model m_LR1, we see that so many variables are not significant on the survival rate. 
m_LR_s<- glm(Survived~. -Fare -Parch, train_data, family = "binomial") 
summary(m_LR_s)

#4. Prediction
head(data_test)
test_data$predicted <- predict(m_LR_s, newdata=test_data, type="response")
summary(test_data$predicted)

#Let's see the distribution of predicted data.
library(ggplot2)
ggplot(test_data, aes(x=predicted)) +
  geom_density(lwd=0.5) +
  labs(title = "Distribution of Probability Prediction Data") +
  theme_minimal()

#From the plot, we can see that the data interpreted the distribution of prediction data lean more towards 0
#which means Not Survived. If we set the threshold is 0.5 (by default)

#If the probability is greater than 0.5, the label will get 1 (survived).
#If the probability is lower than 0.5, the label will get 0 (not survived)

test_data$pred_label <- factor(ifelse(test_data$predicted > 0.5, 1, 0)) #1 = survived, 0 = not survived
test_data %>% 
  select(Survived, pred_label) %>% 
  head()

#5. MODEL EVALUATION

#In this case, the positive class (1) is survived. 
#We'll use sensitivity / recall metric for model evaluation in this case, 
#because we don't wish to see the False Negative class (actually survived but the model predicted not survived).

#For model evaluation, we'll use confusionMatrix() function from package caret
library(caret)
confusionMatrix(reference = as.factor(test_data$Survived), 
                data = as.factor(test_data$pred_label), positive = "1" )

#From the confusion matrix of the prediction using logistic regression model, we can get information such as:
#1. The model can predict correctly in positive class and negative class base on the real data is 77.5% (Accuracy)
#2. The measurement of the model's goodness of the positive class (survived) / Sensitivity is 68.00%
#3. Based on the Specificity metrics, we assume the model is good enough to predict the data test.

#In this case, logistic regression model predicts the test data well. 


#ROC
roc_pred <- prediction(predictions = test_data$predicted  , labels = test_clean$Survived)
roc_perf <- performance(roc_pred , "tpr" , "fpr")
plot(roc_perf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

library(ROSE)
roc.curve(test_clean$Survived, test_data$predicted, add.roc=TRUE, col="red")

## Area under the curve (AUC): 0.869

