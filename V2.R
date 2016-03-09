#---------------------------------------------------------------------------------------
#Step 1. Prepare Data Load libraries, load dataset, check the structore of the data
#-----------------------------------------------------------------------------------
# initialize the packages
library(ipred)
library(e1071)
library(MASS)
library(date)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(corrplot)
library(car)
library(caret)
library(mlbench)
library(glmnet)
library(kernlab)
library(rpart)
library(gbm)
library(caTools)
library(randomForest)
library(Cubist)
library(reshape2)
library(tidyr)

#----------------------------------------
#step1. read file and check the structure
#----------------------------------------
call_cnts <- read.delim("final_dataset.csv", quote="", stringsAsFactors=TRUE)

str(call_cnts)


#-----------------------------------------------
#Step 2. convert the data type. 
#Data Cleaning (remove duplicates, remove irrelevant attributes, Impute missing data)
#----------------------------------------------------

call_cnts$call_date <- as.Date(call_cnts$call_date)
#change the data type
call_cnts$calendar_quarter <- factor(call_cnts$calendar_quarter)
call_cnts$calendar_month_number <- factor(call_cnts$calendar_month_number)
call_cnts$week_day_number <- factor(call_cnts$week_day_number)
call_cnts$Working_day <- factor(call_cnts$Working_day)
call_cnts$is_a_weekend_day <- factor(call_cnts$is_a_weekend_day)
call_cnts$is_a_public_holiday <- factor(call_cnts$is_a_public_holiday)
call_cnts$season <- factor(call_cnts$season, labels = c("Winter", "Spring","Summer", "Autumn"))


call_cnts$date_key  <- NULL
call_cnts$wevents <- NULL

#Check for nulls and add replace missing value with median
table(is.na(call_cnts))
colSums(is.na(call_cnts))
table(is.na(call_cnts$Temp))
call_cnts$Temp[is.na(call_cnts$Temp)] <- median(call_cnts$Temp, na.rm = TRUE)

#-------------------------------------------------------------------
#Step 3. Perform data-split to get training /validation dataset
#split the data 80% train/20% test
#------------------------------------------------------------------

set.seed(849)

sample_idx <- createDataPartition(call_cnts$call_cnt, p = 0.8, list=FALSE, times=1)
data_train <- call_cnts[sample_idx, ]
data_test  <- call_cnts[-sample_idx, ]

str(data_train)
str(data_test)

#-------------------------------------------------------------------
#Step 4. Summarize Data
#Descriptive statistics (summary, class distributions,skew)
#------------------------------------------------------------------
summary(data_train)
fivenum(data_train$call_cnt)
table(data_train$call_city)
table(data_train$weather_Events)
table(data_train$Month_name)

#---------------------------------------------------------------------------------------------------
#Step 5. Data visualizations (histograms, boxplots, correlation plots, scatterplot matrix ('pairs'))
#---------------------------------------------------------------------------------------------------
# Histogram for all the Variables in one graph
mpgid <- mutate(data_train, id=as.numeric(rownames(data_train)))
mpgstack <- melt(mpgid, id="id")
pp <- qplot(value, data=mpgstack) + facet_wrap(~variable, scales="free")
ggsave("mpg-histograms.pdf", pp, scale=2)

#Correlation Plots are done before changing the numeric datatypes to factors
cor_cnt <- cor(data_train[sapply(data_train, is.numeric)])
corrplot(cor_cnt, method = "circle", order = "alphabet")

#boxplot
a <- ggplot(data = data_train, aes(x=weather_Events, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
b <- ggplot(data = data_train, aes(x=Working_day, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
c <- ggplot(data = data_train, aes(x=week_day_name, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
d <- ggplot(data = data_train, aes(x=call_city, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
e <- ggplot(data = data_train, aes(x=Month_name, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
f <- ggplot(data = data_train, aes(x=is_a_public_holiday, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
g <- ggplot(data = data_train, aes(x=season, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
h <- ggplot(data = data_train, aes(x=Temp, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
#multiple graphs in one graph
grid.arrange(a, g,  ncol=2, nrow =1)
grid.arrange( b, f, c,  e, ncol=2, nrow =2)
d

#Scatterplots
scatterplotdata <- data_train[ ,c("call_cnt","call_city","call_date","Temp", "Working_day", "is_a_public_holiday", "weather_Events", "season")]
#plot(scatterplotdata)
# scatterplotmatrix with  histogram along the diagonal
scatterplotMatrix(scatterplotdata,
                  diagonal="histogram", smooth=FALSE)

#---------------------------------------------------------------------------------------------------
#Step 6. 
#Evaluate Algorithms, Model Training.
#Choose options to estimate model error (e.g. cross validation); 
#choose evaluation metric (RMSE, R2 etc.)
#Train different algorithms (estimate accuracy of several algorithms on training set)
#Compare Algorithms (based on chosen metric of accuracy)
#---------------------------------------------------------------------------------------------------



#set the controls
#cctrlcv5 <- trainControl(method = "cv", number = 5)
cctrlcv10 <- trainControl(method = "cv", number = 10, returnResamp = "all")
#cctrlrp <- trainControl(method='repeatedcv',number=10,repeats=3)

formula <- call_cnt ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+week_day_name+weather_Events+season

#Models with no tranformation
#lm
set.seed(849)
fit.lm <- train(call_cnt ~., data=data_train, method="lm", metric="RMSE", 
                preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.lm)

# glmnet evaluation
set.seed(849)
fit.glmnet <- train(call_cnt ~., data=data_train, method="glmnet", metric="RMSE", 
                    preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.glmnet)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knn <- train(formula, data=data_train, method="knn", metric="RMSE", 
                 preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.knn)

#SVM Radial Model
set.seed(849)
fit.svmRadial <- train(formula, data=data_train, method="svmRadial", metric="RMSE",preProc = c("center", "scale"),
                       trControl=cctrlcv10)
print(fit.svmRadial)

# CART Model
set.seed(849)
fit.rpart <- train(formula, data=data_train, method="rpart", metric="RMSE", preProc = c("center", "scale"),
                   trControl=cctrlcv10)
print(fit.rpart)

# GBM Model
set.seed(849)
fit.gbm <- train(call_cnt ~., data=data_train, method = "gbm",metric="RMSE",preProc = c("center", "scale"),
                 verbose = FALSE,   trControl = cctrlcv10 )
print(fit.gbm)

# Random forest Model
set.seed(849)
fit.rf <- train(call_cnt ~., data = data_train,method = "rf",metric="RMSE",preProc = c("center", "scale"),
                verbose = FALSE,    trControl = cctrlcv10 )
print(fit.rf)

#Model cubist
set.seed(849)
fit.cubist <- train(call_cnt ~., data = data_train,method = "cubist",metric="RMSE",preProc = c("center", "scale"),
                    verbose = FALSE,Control = cctrlcv10 )
print(fit.cubist)


#Model treebag
set.seed(849)
fit.treebag <- train(call_cnt ~., data = data_train,method = "treebag",preProc = c("center", "scale"),
                     verbose = FALSE,Control = cctrlcv10 )
print(fit.treebag)

#Compare all the models and Visualize the RMSE differences
cvValues <- resamples(list(glmnet = fit.glmnet , CART = fit.rpart, SVM = fit.svmRadial, lm = fit.lm, knn = fit.knn ))
summary(cvValues)
bwplot(cvValues, metric = "RMSE")

ensemblevalues1 <- resamples(list(gbm = fit.gbm, rf = fit.rf ))
summary(ensemblevalues1)
bwplot(ensemblevalues1, metric = "RMSE")

ensemblevalues2 <- resamples(list(cubist = fit.cubist, treebag = fit.treebag ))
summary(ensemblevalues2)
bwplot(ensemblevalues2, metric = "RMSE")

#------------------------------------------------
#Models with BoxCox transformation on X variables
str(data_train)

#Preprocess and transform
preProcValues <- c("center", "scale", "BoxCox")

#transform the varibales using BoxCox and check the distributions of each variable 
data_train_hist <- predict(preProcess(data_train, method=preProcValues), data_train)

# Histogram after the transformation for all the Variables in one graph
mpgid <- mutate(data_train_hist, id=as.numeric(rownames(data_train_hist)))
mpgstack <- melt(mpgid, id="id")
pp <- qplot(value, data=mpgstack) + facet_wrap(~variable, scales="free")

#formual for building models
formulaboxx <- call_cnt ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+week_day_name

# lm evaluation
set.seed(849)
fit.lmtransx <- train(formulaboxx, data=data_train, method="lm", metric="RMSE", preProc=preProcValues,
                      trControl=cctrlcv10)
print(fit.lmtransx)

# glmnet evaluation
set.seed(849)
fit.glmnettransx <- train(formulaboxx, data=data_train, method="glmnet", metric="RMSE", preProc=preProcValues,
                          trControl=cctrlcv10)
print(fit.glmnettransx)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knntransx <- train(formulaboxx, data=data_train, method="knn", metric="RMSE", preProc=preProcValues,
                       trControl=cctrlcv10)
print(fit.knntransx)

#SVM Radial Model
set.seed(849)
fit.svmRadialtransx <- train(formulaboxx, data=data_train, method="svmRadial", metric="RMSE",preProc=preProcValues,
                            trControl=cctrlcv10)
print(fit.svmRadialtransx)

# CART Model
set.seed(849)
fit.rparttransx <- train(formulaboxx, data=data_train, method="rpart", metric="RMSE", preProc=preProcValues,
                        trControl=cctrlcv10)
print(fit.rparttransx)


# GBM Model
set.seed(849)
fit.gbmtransx <- train(formulaboxx, data=data_train, method = "gbm",metric="RMSE",preProc=preProcValues,
                      verbose = FALSE,   trControl = cctrlcv10 )
print(fit.gbmtransx)


# Random forest Model
set.seed(849)
fit.rftransx <- train(formulaboxx, data=data_train,method = "rf",metric="RMSE",preProc=preProcValues,
                     verbose = FALSE,    trControl = cctrlcv10)
print(fit.rftransx)


#Model cubist
set.seed(849)
fit.cubisttransx <- train(formulaboxx, data=data_train,method = "cubist",metric="RMSE",preProc=preProcValues,
                         verbose = FALSE,Control = cctrlcv10 )
print(fit.cubisttransx)


#Model treebag
set.seed(849)
fit.treebagtransx <- train(formulaboxx, data=data_train,method = "treebag",preProc=preProcValues,
                          verbose = FALSE,Control = cctrlcv10 )
print(fit.treebagtransx)


#Compare Models and visualize the results
cvValuestransx <- resamples(list(glmnet = fit.glmnettransx , CART = fit.rparttransx, SVM = fit.svmRadialtransx, lm = fit.lmtransx, knn = fit.knntransx ))

summary(cvValuestransx)
bwplot(cvValuestransx, metric = "RMSE")


ensemblevaluestrans1 <- resamples(list(gbm = fit.gbmtransx, rf = fit.rftransx  ))
summary(ensemblevaluestrans1)
bwplot(ensemblevaluestrans1, metric = "RMSE")

ensemblevaluestrans2 <- resamples(list(cubist = fit.cubisttransx, treebag = fit.treebagtransx ))
summary(ensemblevaluestrans2)
bwplot(ensemblevaluestrans2, metric = "RMSE")



#-------------------------------------------------------
#Models with tuning parameters
#-------------------------------------------------------
# tuned model for svmradial with tunelength =10
set.seed(849)
fit.svmRadial2 <- train(formula, data = data_train, method = "svmRadial",metric="RMSE",
                        preProcess = c("center", "scale"), tuneLength = 10,trControl = cctrlcv10)
#summary
print(fit.svmRadial2)

# tune glmnet with tunelength=10
set.seed(849)

fit.glmnet2 <- train(formula, data = data_train,method = "glmnet",metric="RMSE",
                        preProcess = c("center", "scale"),tuneLength = 10,trControl = cctrlcv10)

print(fit.glmnet2)

# tune glmnet with tunelength=10
set.seed(849)

fit.rf2 <- train(formula, data = data_train,method = "rf",metric="RMSE",
                     preProcess = c("center", "scale"),tuneLength = 10,trControl = cctrlcv10)

print(fit.rf2)


#comparing tuned model
cvValuestune <- resamples(list(glmnet = fit.glmnet2 , SVM = fit.svmRadial2, rf = fit.rf2 ))
summary(cvValuestune)

#visualizing the results
bwplot(cvValuestune, metric = "RMSE")

#------------------------------------------------
#Models with Log transformation on Y variable
#-------------------------------------------------
formulalog <- log(call_cnt) ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+week_day_name
# lm evaluation
set.seed(849)
fit.lmlog <- train(formulalog, data=data_train, method="lm", metric="RMSE", 
                     preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.lmlog)

# glmnet evaluation
set.seed(849)
fit.glmnetlog <- train(formulalog, data=data_train, method="glmnet", metric="RMSE", preProc=preProcValues,
                          trControl=cctrlcv10)
print(fit.glmnetlog)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knnlog <- train(formulaboxx, data=data_train, method="knn", metric="RMSE", preProc=preProcValues,
                       trControl=cctrlcv10)
print(fit.knnlog)

#SVM Radial Model
set.seed(849)
fit.svmRadiallog <- train(formulalog, data=data_train, method="svmRadial", metric="RMSE",preProc=preProcValues,
                             trControl=cctrlcv10)
print(fit.svmRadiallog)

# CART Model
set.seed(849)
fit.rpartlog <- train(formulalog, data=data_train, method="rpart", metric="RMSE", preProc=preProcValues,
                         trControl=cctrlcv10)
print(fit.rpartlog)


# GBM Model
set.seed(849)
fit.gbmlog <- train(formulalog, data=data_train, method = "gbm",metric="RMSE",preProc=preProcValues,
                       verbose = FALSE,   trControl = cctrlcv10 )
print(fit.gbmlog)


# Random forest Model
set.seed(849)
fit.rflog <- train(formulalog, data=data_train,method = "rf",metric="RMSE",preProc=preProcValues,
                      verbose = FALSE,    trControl = cctrlcv10)
print(fit.rflog)


#Model cubist
set.seed(849)
fit.cubistlog <- train(formulalog, data=data_train,method = "cubist",metric="RMSE",preProc=preProcValues,
                          verbose = FALSE,Control = cctrlcv10 )
print(fit.cubistlog)


#Model treebag
set.seed(849)
fit.treebaglog <- train(formulalog, data=data_train,method = "treebag",preProc=preProcValues,
                           verbose = FALSE,Control = cctrlcv10 )
print(fit.treebaglog)

#Compare Models and visualize the results
cvValueslog <- resamples(list(glmnet = fit.glmnetlog , CART = fit.rpartlog, SVM = fit.svmRadiallog, lm = fit.lmlog, knn = fit.knnlog ))

summary(cvValueslog)
bwplot(cvValueslog, metric = "RMSE")


ensemblevalueslog1 <- resamples(list(gbm = fit.gbmlog, rf = fit.rflog  ))
summary(ensemblevalueslog1)
bwplot(ensemblevalueslog1, metric = "RMSE")

ensemblevalueslog2 <- resamples(list(cubist = fit.cubistlog, treebag = fit.treebaglog ))
summary(ensemblevalueslog2)
bwplot(ensemblevalueslog2, metric = "RMSE")

#---------------------------------------------------------------------------------------------------
#Step 7. 
#Model Evaluation
#Finalize Model and estimate error
#Use the best algorithm(s) to predict on validation dataset and estimate error on unseen data.
#---------------------------------------------------------------------------------------------------

fit.lm$results
fit.glmnetlog$results
fit.glmnet2$results
fit.lmtransx$results

#variable importance Grapgh
plot(varImp(fit.glmnetlog))

#-----------------------------------------------------------------------------------------

#Predict Values using the fit.lm model
print(fit.lm)
predicted_callcnt_lm<- predict(fit.lm,data_test)
plot(predicted_callcnt_lm,data_test$call_cnt)
fit.lm.RMSE <- RMSE(predicted_callcnt_lm,data_test$call_cnt)
fit.lm.R2 <- R2(predicted_callcnt_lm,data_test$call_cnt)
predictmodelvalues_lm<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_lm)
head(predictmodelvalues_lm)

#Predict Values using fit.glmnetlog Model
predicted_callcnt_log<- exp(predict(fit.glmnetlog,data_test))
plot(predicted_callcnt_log,data_test$call_cnt, color = "blue", main = "Predicted vs. Actual")
fit.glmnetlog.RMSE <- RMSE(predicted_callcnt_log,data_test$call_cnt,na.rm=TRUE)
fit.glmnetlog.R2 <- R2(predicted_callcnt_log,data_test$call_cnt,na.rm=TRUE)
predictmodelvalues_log<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_log)
head(predictmodelvalues_log)

#Predict Values using the Model that is tuned with tunelength=10
predicted_callcnt_tune<- (predict(fit.glmnet2,data_test))^0.4
plot(predicted_callcnt_tune,data_test$call_cnt)
fit.glmnet2.RMSE <- RMSE(predicted_callcnt_tune,data_test$call_cnt, na.rm=TRUE)
fit.glmnet2.R2 <- R2(predicted_callcnt_tune,data_test$call_cnt, na.rm=TRUE)
predictmodelvalues_tune<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_tune)
head(predictmodelvalues_tune)


#Predict Values using the fit.lmtransx model
print(fit.lmtransx)
predicted_callcnt_lmtransx<- predict(fit.lm,data_test)
#plot(predicted_callcnt_lmtransx,data_test$call_cnt)
fit.lmtransx.RMSE <- RMSE(predicted_callcnt_lmtransx,data_test$call_cnt)
fit.lmtransx.R2 <- R2(predicted_callcnt_lmtransx,data_test$call_cnt)
predictmodelvalues_lmtransx<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_lmtransx)
head(predictmodelvalues_lmtransx)


#---------------------------------------------
#step8.  put into all together with results
#--------------------------------------------------

# Create a data frame with the error metrics for each method
results <- data.frame(Method = c("Linear Regression","glmnet with log","glmnettune", "lmtransx"),
                           RMSE   = c(fit.lm.RMSE, fit.glmnetlog.RMSE, fit.glmnet2.RMSE, fit.lmtransx.RMSE),
                           R2    = c(fit.lm.R2, fit.glmnetlog.R2, fit.glmnet2.R2, fit.lmtransx.R2)) 

# Round the values and print the table
results$RMSE <- round(results$RMSE,2)
results$R2 <- round(results$R2,2) 

results


#Visualze the results for the Actual and predictable values

ggplot(data = predictmodelvalues_lm,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for Linear regression")



ggplot(data = predictmodelvalues_log,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for glmnet Log")



ggplot(data = predictmodelvalues_tune,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for glmnet tunelength")


ggplot(data = predictmodelvalues_lmtransx,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for Linear regression with BoxCox transformation")

























