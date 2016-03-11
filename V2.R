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
call_cnts$calendar_month_number <- factor(call_cnts$calendar_month_number, labels = c("Jan", "Feb","Mar", "Apr", "May", "Jun","Jul", "Aug", "Sept", "Oct", "Nov", "Dec"))
call_cnts$Working_day <- factor(call_cnts$Working_day)
call_cnts$is_a_public_holiday <- factor(call_cnts$is_a_public_holiday)
call_cnts$season <- factor(call_cnts$season, labels = c("Winter", "Spring","Summer", "Autumn"))

call_cnts$date_key  <- NULL
call_cnts$wevents <- NULL
call_cnts$Month_name <- NULL
call_cnts$week_day_name <- NULL
call_cnts$week_day_number <- NULL
call_cnts$is_a_weekend_day <- NULL

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
table(data_train$calendar_month_number)
table(data_train$season)

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
c <- ggplot(data = data_train, aes(x=calendar_quarter, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
d <- ggplot(data = subset(data_train, call_city == 'HAMILTON'), aes(x=Month_name, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
e <- ggplot(data = data_train, aes(x=calendar_month_number, y=call_cnt))+geom_boxplot(fill="chartreuse4") +coord_flip()
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

#formula <- call_cnt ~ call_city+calendar_month_number+is_a_public_holiday+Working_day+Humidity

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
fit.knn <- train(call_cnt ~., data=data_train, method="knn", metric="RMSE", 
                 preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.knn)

#SVM Radial Model
set.seed(849)
fit.svmRadial <- train(call_cnt ~., data=data_train, method="svmRadial", metric="RMSE",preProc = c("center", "scale"),
                       trControl=cctrlcv10)
print(fit.svmRadial)

# CART Model
set.seed(849)
fit.rpart <- train(call_cnt ~., data=data_train, method="rpart", metric="RMSE", preProc = c("center", "scale"),
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

ensemblevalues1 <- resamples(list(gbm = fit.gbm, rf = fit.rf))
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
formulaboxx <- call_cnt ~ call_city+call_date+Temp+Humidity+Windspeed+is_a_public_holiday+Working_day

# lm evaluation
set.seed(849)
fit.lmtransx <- train(call_cnt ~., data=data_train, method="lm", metric="RMSE", preProc=preProcValues,
                      trControl=cctrlcv10)
print(fit.lmtransx)

# glmnet evaluation
set.seed(849)
fit.glmnettransx <- train(call_cnt ~., data=data_train, method="glmnet", metric="RMSE", preProc=preProcValues,
                          trControl=cctrlcv10)
print(fit.glmnettransx)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knntransx <- train(call_cnt ~., data=data_train, method="knn", metric="RMSE", preProc=preProcValues,
                       trControl=cctrlcv10)
print(fit.knntransx)

#SVM Radial Model
set.seed(849)
fit.svmRadialtransx <- train(call_cnt ~., data=data_train, method="svmRadial", metric="RMSE",preProc=preProcValues,
                            trControl=cctrlcv10)
print(fit.svmRadialtransx)

# CART Model
set.seed(849)
fit.rparttransx <- train(call_cnt ~., data=data_train, method="rpart", metric="RMSE", preProc=preProcValues,
                        trControl=cctrlcv10)
print(fit.rparttransx)


# GBM Model
set.seed(849)
fit.gbmtransx <- train(call_cnt ~., data=data_train, method = "gbm",metric="RMSE",preProc=preProcValues,
                      verbose = FALSE,   trControl = cctrlcv10 )
print(fit.gbmtransx)


# Random forest Model
set.seed(849)
fit.rftransx <- train(call_cnt ~., data=data_train,method = "rf",metric="RMSE",preProc=preProcValues,
                     verbose = FALSE,    trControl = cctrlcv10)
print(fit.rftransx)


#Model cubist
set.seed(849)
fit.cubisttransx <- train(call_cnt ~., data=data_train,method = "cubist",metric="RMSE",preProc=preProcValues,
                         verbose = FALSE,Control = cctrlcv10 )
print(fit.cubisttransx)


#Model treebag
set.seed(849)
fit.treebagtransx <- train(call_cnt ~., data=data_train,method = "treebag",preProc=preProcValues,
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
fit.cubist2 <- train(call_cnt~., data = data_train, method = "cubist",metric="RMSE",
                        preProcess = c("center", "scale", "BoxCox"), tuneLength = 10,trControl = cctrlcv10)
#summary
print(fit.cubist2)

set.seed(849)
fit.gbm2 <- train(call_cnt~., data = data_train, method = "gbm",metric="RMSE",
                     preProcess = c("center", "scale", "BoxCox"), tuneLength = 10,trControl = cctrlcv10)
#summary
print(fit.gbm2)

# tune glmnet with tunelength=10
set.seed(849)

fit.glmnet2 <- train(call_cnt~., data = data_train,method = "glmnet",metric="RMSE",
                        preProcess = c("center", "scale", "BoxCox"),tuneLength = 10,trControl = cctrlcv10)

print(fit.glmnet2)

# tune glmnet with tunelength=10
set.seed(849)

fit.rf2 <- train(call_cnt~., data = data_train,method = "rf",metric="RMSE",
                     preProcess = c("center", "scale", "BoxCox"),tuneLength = 10,trControl = cctrlcv10)

print(fit.rf2)


#comparing tuned model
cvValuestune <- resamples(list(glmnet = fit.glmnet2 , cubist = fit.cubist2, rf = fit.rf2, gbm = fit.gbm2 ))
summary(cvValuestune)

#visualizing the results
bwplot(cvValuestune, metric = "RMSE")

#---------------------------------------------------------------------------------------------------
#Step 7. 
#Model Evaluation
#Finalize Model and estimate error
#Use the best algorithm(s) to predict on validation dataset and estimate error on unseen data.
#---------------------------------------------------------------------------------------------------

fit.glmnet2$results
fit.gbm2$results
fit.cubist2$results
fit.rf2$results

#-----------------------------------------------------------------------------------------

#Predict Values using the glmnet
print(fit.glmnet2)
predicted_callcnt_glmnet<- predict(fit.glmnet2,data_test)
#plot(predicted_callcnt_glmnet,data_test$call_cnt)
fit.glment.RMSE <- RMSE(predicted_callcnt_glmnet,data_test$call_cnt)
fit.glment.R2 <- R2(predicted_callcnt_glmnet,data_test$call_cnt)
predictmodelvalues_glmnet<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_glmnet)
head(predictmodelvalues_glmnet)

#Predict Values using gbm Model
predicted_callcnt_gbm<- predict(fit.gbm2,data_test)
#plot(predicted_callcnt_gbm,data_test$call_cnt, color = "blue", main = "Predicted vs. Actual")
fit.gbm.RMSE <- RMSE(predicted_callcnt_log,data_test$call_cnt,na.rm=TRUE)
fit.gbm.R2 <- R2(predicted_callcnt_gbm,data_test$call_cnt,na.rm=TRUE)
predictmodelvalues_gbm<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_gbm)
head(predictmodelvalues_gbm)

#Predict Values using the Model cubist
predicted_callcnt_cubist<- predict(fit.cubist2,data_test)
plot(predicted_callcnt_cubist,data_test$call_cnt)
fit.cubist.RMSE <- RMSE(predicted_callcnt_cubist,data_test$call_cnt, na.rm=TRUE)
fit.cubist.R2 <- R2(predicted_callcnt_cubist,data_test$call_cnt, na.rm=TRUE)
predictmodelvalues_cubist<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_cubist)
head(predictmodelvalues_cubist)

#Predict Values using the Model rf
predicted_callcnt_rf<- predict(fit.rf2,data_test)
plot(predicted_callcnt_rf,data_test$call_cnt)
fit.rf.RMSE <- RMSE(predicted_callcnt_rf,data_test$call_cnt, na.rm=TRUE)
fit.rf.R2 <- R2(predicted_callcnt_rf,data_test$call_cnt, na.rm=TRUE)
predictmodelvalues_rf<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt_rf)
head(predictmodelvalues_rf)

#---------------------------------------------
#step8.  put into all together with results
#--------------------------------------------------

# Create a data frame with the error metrics for each method
results <- data.frame(Method = c("gbm","rf","cubist", "glmnet"),
                           RMSE   = c(fit.gbm.RMSE, fit.rf.RMSE, fit.cubist.RMSE, fit.glment.RMSE),
                           R2    = c(fit.gbm.R2, fit.rf.R2, fit.cubist.R2, fit.glment.R2)) 

# Round the values and print the table
results$RMSE <- round(results$RMSE,2)
results$R2 <- round(results$R2,2) 

results


#Visualze the results for the Actual and predictable values

ggplot(data = predictmodelvalues_gbm,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for gbm")



ggplot(data = predictmodelvalues_rf,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for random forest")



ggplot(data = predictmodelvalues_glmnet,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for glmnet ")


ggplot(data = predictmodelvalues_cubist,aes(x = actual , y = predicted )) + 
  geom_point(colour = "blue") +    ggtitle("Predicted vs. Actual for cubist")
















