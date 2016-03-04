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
library(lmtest) # bptest for testing heteroscedasticity
library(gvlma) # global validation of linear model assumptions


call_cnts <- read.delim("final_dataset.csv", quote="", stringsAsFactors=TRUE)

str(call_cnts)
#Check for nulls and add replace missing value with median
table(is.na(call_cnts))
colSums(is.na(call_cnts))
table(is.na(call_cnts$Temp))
call_cnts$Temp[is.na(call_cnts$Temp)] <- median(call_cnts$Temp, na.rm = TRUE)

#-----------------------------------------------
#Step 2. convert the data type. 
#Data Cleaning (remove duplicates, remove irrelevant attributes, impute/delete missing data)
#Feature Selection (remove redundant features)
#----------------------------------------------------
#Correlation Plots are done before changing the numeric datatypes to factors
cor_cnt <- cor(call_cnts[sapply(call_cnts, is.numeric)])
corrplot(cor_cnt, method = "circle", order = "alphabet")

#Convert the datatype 
call_cnts$call_date <- as.Date(call_cnts$call_date)
call_cnts$calendar_quarter <- factor(call_cnts$calendar_quarter)
#call_cnts$calendar_month_number <- factor(call_cnts$calendar_month_number)
#call_cnts$week_day_number <- factor(call_cnts$week_day_number)
call_cnts$Working_day <- factor(call_cnts$Working_day)
call_cnts$is_a_weekend_day <- factor(call_cnts$is_a_weekend_day)
call_cnts$is_a_public_holiday <- factor(call_cnts$is_a_public_holiday)
call_cnts$season <- factor(call_cnts$season, labels = c("Winter", "Spring","Summer", "Autumn"))
call_cnts$wevents <- factor(call_cnts$wevents, labels = c("Normal", "Rain","Fog", "Snow", "Thunderstorm", "Rain-Snow","Fog_Rain", "Fog_R_Snow"))

#remove the duplicate columns
call_cnts$calendar_month_number <- NULL
call_cnts$week_day_number   <- NULL
call_cnts$date_key  <- NULL
#call_cnts$weather_Events  <- NULL

#-------------------------------------------------------------------
#Step 3. Perform data-split to get training /validation dataset
#split the data 80% train/20% test
#------------------------------------------------------------------

set.seed(849)
sample_idx <- sample(nrow(call_cnts), nrow(call_cnts)*0.8)
data_train <- call_cnts[sample_idx, ]
data_test <- call_cnts[-sample_idx, ]

str(data_train)
str(data_test)

#--------------------------------------------------------------------------
#Tried to use this in my Train function for x and y variables but did not work. 
#so Manually added the variables.
#--------------------------------------------------------------------
#generalize the predicator and outcome variables
outcome <- 'call_cnt'
#predicators <- names(call_cnts)[names(call_cnts) != outcome]
predicators <- c("call_city", "call_date" , "Temp" ,"Humidity", "Windspeed" ,"calendar_quarter", "Month_name" ,"weather_Events",
                 "week_day_name","Working_day", "is_a_weekend_day"  ,  "is_a_public_holiday" ,"season"  )             

y <- data_train[ ,outcome]
x <- data_train[ ,predicators]                

#-------------------------------------------------------------------
#Step 4. Summarize Data
#Descriptive statistics (summary, class distributions, standard deviations, skew)
#------------------------------------------------------------------
summary(data_train)
fivenum(data_train$call_cnt)

#---------------------------------------------------------------------------------------------------
#Step 5. Data visualizations (histograms, boxplots, correlation plots, scatterplot matrix ('pairs'))
#---------------------------------------------------------------------------------------------------

# Histogram for all the Variables in one graph
mpgid <- mutate(data_train, id=as.numeric(rownames(data_train)))
mpgstack <- melt(mpgid, id="id")
pp <- qplot(value, data=mpgstack) + facet_wrap(~variable, scales="free")
ggsave("mpg-histograms.pdf", pp, scale=2)

#boxplot
a <- ggplot(data = data_train, aes(x=weather_Events, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
b <- ggplot(data = data_train, aes(x=Working_day, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
c <- ggplot(data = data_train, aes(x=week_day_name, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
d <- ggplot(data = data_train, aes(x=call_city, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
e <- ggplot(data = data_train, aes(x=Month_name, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
f <- ggplot(data = data_train, aes(x=is_a_public_holiday, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
g <- ggplot(data = data_train, aes(x=season, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
h <- ggplot(data = data_train, aes(x=Temp, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
#multiple graphs in one graph
grid.arrange(a, g,  ncol=2, nrow =1)
grid.arrange( b, f, c,  e, ncol=2, nrow =2)
d

#Scatterplots
scatterplotdata <- data_train[ ,c("call_cnt","call_city","call_date","Temp", "Working_day", "is_a_public_holiday", "wevents", "season")]
plot(scatterplotdata)

# scatterplotmatrix with  histogram along the diagonal
scatterplotMatrix(scatterplotdata,
                  diagonal="histogram", smooth=FALSE)

#---------------------------------------------------------------------------------------------------
#Step 6. 
#Evaluate Algorithms, Model Training.
#Choose options to estimate model error (e.g. cross validation); choose evaluation metric (RMSE, R2 etc.)
#Train different algorithms (estimate accuracy of several algorithms on training set)
#Compare Algorithms (based on chosen metric of accuracy)
#---------------------------------------------------------------------------------------------------
#Baseline model - predict the mean of the training data
#baseline_mean <- mean(data_train$call_cnt)
#Evaluate RMSE and MAE on the testing dat
#baseline_RMSE <- sqrt(mean((baseline_mean-data_train$call_cnt)^2))
#baseline_MAE <- mean(abs(baseline_mean-data_test$call_cnt))
#-------------------------------------------------------------------
#set the controls
cctrlcv5 <- trainControl(method = "cv", number = 5)
cctrlcv10 <- trainControl(method = "cv", number = 10, returnResamp = "all")
cctrlrp <- trainControl(method='repeatedcv',number=10,repeats=3)

#Models with no tranformation
#lm
set.seed(849)
fit.lm <- train(call_cnt~., data=data_train, method="lm", metric="RMSE", 
                preProc=c("center", "scale"), trControl=cctrlrp)
print(fit.lm)

# GBM Model
set.seed(849)
fit.gbm <- train(call_cnt~., data=data_train, method = "gbm",metric="RMSE",
                 verbose = FALSE,   trControl = cctrlcv5 )
print(fit.gbm)

# Random forest Model
set.seed(849)
fit.rf <- train(call_cnt~ ., data = data_train,method = "rf",metric="RMSE",
                verbose = FALSE,    trControl = cctrlcv5 )
print(fit.rf)

#Model cubist
set.seed(849)
fit.cubist <- train(call_cnt~ ., data = data_train,method = "cubist",metric="RMSE",
                    verbose = FALSE,Control = cctrlcv5 )
print(fit.cubist)


#Model treebag
set.seed(849)
fit.treebag <- train(call_cnt~ ., data = data_train,method = "treebag",
                     verbose = FALSE,Control = cctrlcv5 )
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


#BoxCox transformation
------------------------------
#transcallcnt <- BoxCoxTrans(data_train$call_cnt)
callcnt_new     <- predict(BoxCoxTrans(data_train$call_cnt), data_train$call_cnt)
Temp_new        <- predict(BoxCoxTrans(data_train$Temp), data_train$Temp)
Windspeed_new   <- predict(BoxCoxTrans(data_train$Windspeed), data_train$Windspeed)
Humidity_new    <- predict(BoxCoxTrans(data_train$Humidity), data_train$Humidity)

# append the transformed variable to data_train
data_train <- cbind(data_train, Temp_new, Windspeed_new, Humidity_new,callcnt_new)

# Histogram after the transformation for all the Variables in one graph
mpgid <- mutate(data_train, id=as.numeric(rownames(data_train)))
mpgstack <- melt(mpgid, id="id")
pp <- qplot(value, data=mpgstack) + facet_wrap(~variable, scales="free")
ggsave("mpg-histograms_after_transform.pdf", pp, scale=2)

head(data_train,5) # view the top 6 rows

#------------------------------------------------
#Models with BoxCox transformation on Y variable
#-------------------------------------------------
# lm evaluation
set.seed(849)
fit.lmtrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train, method="lm", metric="RMSE", 
                preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.lmtrans)


# glmnet evaluation
set.seed(849)
fit.glmnettrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train, method="glmnet", metric="RMSE", 
                    preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.glmnettrans)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knntrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train, method="knn", metric="RMSE", 
                 preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.knntrans)

#SVM Radial Model
set.seed(849)
fit.svmRadialtrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train, method="svmRadial", metric="RMSE",
                       trControl=cctrlcv10)
print(fit.svmRadialtrans)

# CART Model
set.seed(849)
fit.rparttrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train, method="rpart", metric="RMSE", 
                   trControl=cctrlcv10)
print(fit.rparttrans)


# GBM Model
set.seed(849)
fit.gbmtrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data = data_train, method = "gbm",metric="RMSE",
                 verbose = FALSE,   trControl = cctrlcv5 )
print(fit.gbmtrans)


# Random forest Model
set.seed(849)
fit.rftrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data = data_train,method = "rf",metric="RMSE",
                verbose = FALSE,    trControl = cctrlcv5 )
print(fit.rftrans)


#Model cubist
set.seed(849)
fit.cubisttrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data = data_train,method = "cubist",metric="RMSE",
                    verbose = FALSE,Control = cctrlcv5 )
print(fit.cubisttrans)


#Model treebag
set.seed(849)
fit.treebagtrans <- train(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data = data_train,method = "treebag",
                     verbose = FALSE,Control = cctrlcv5 )
print(fit.treebagtrans)


#Compare Models and visualize the results
cvValuestrans <- resamples(list(glmnet = fit.glmnettrans , CART = fit.rparttrans, SVM = fit.svmRadialtrans, lm = fit.lmtrans, knn = fit.knntrans ))

summary(cvValuestrans)
bwplot(cvValuestrans, metric = "RMSE")


ensemblevaluestrans1 <- resamples(list(gbm = fit.gbmtrans, rf = fit.rftrans ))
summary(ensemblevaluestrans1)
bwplot(ensemblevaluestrans1, metric = "RMSE")

ensemblevaluestrans2 <- resamples(list(cubist = fit.cubisttrans, treebag = fit.treebagtrans ))
summary(ensemblevaluestrans2)
bwplot(ensemblevaluestrans2, metric = "RMSE")

#-------------------------------------------------------
#Models with tuning parameters
#-------------------------------------------------------
# tuned model for svmradial with tunelength =10
set.seed(849)
fit.svmRadial2 <- train(call_cnt ~ ., data = data_train, method = "svmRadial",metric="RMSE",
                        preProcess = c("center", "scale"), tuneLength = 10,trControl = cctrlcv10)
#summary
print(fit.svmRadial2)

# tune glmnet with tunelength=10
set.seed(849)

fit.glmnet2 <- train(call_cnt ~ ., data = data_train,method = "glmnet",metric="RMSE",
                        preProcess = c("center", "scale"),tuneLength = 10,trControl = cctrlcv10)

print(fit.glmnet2)

#comparing tuned model
cvValuestune <- resamples(list(glmnet = fit.glmnet2 , SVM = fit.svmRadial2 ))
summary(cvValuestune)

#visualizing the results
bwplot(cvValuestune, metric = "RMSE")

#----------------------------------------------------
#tunegrid
#------------------------------------------------------
set.seed(849)
lambda.seq <- 0.4
fit.glmnet3  <- train(call_cnt ~ ., data = data_train,method ="glmnet",preProc=c('center','scale'),
                      trControl=cctrlrp, tuneGrid = expand.grid(alpha = 1, lambda = 0.4))
print(fit.glmnet3)

# train the model svmradial
set.seed(849)
fit.svmradial3 <- train(call_cnt ~ ., data = data_train,
                        method = "svmRadial",
                        tuneGrid = data.frame(.C = c(1, .5, 2),
                                              .sigma = .01), 
                        trControl = cctrlrp,
                        preProc = c("center", "scale"))

print(fit.svmradial3)

#comparing tuned model and visualize the results
cvValuestunegrid <- resamples(list(glmnet = fit.glmnet3 , SVM = fit.svmradial3 ))
summary(cvValuestunegrid)

#visualizing the results
dotplot(cvValuestunegrid, metric = "RMSE")
bwplot(cvValuestunegrid, metric = "RMSE")

#tune gbm
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)
nrow(gbmGrid)

set.seed(849)
fit.gbm2 <- train(call_cnt ~ ., data = data_train,
                  method = "gbm",
                  trControl = cctrlcv5,
                  verbose = FALSE,
                  metric = "RMSE",
                  ## Now specify the exact models 
                  ## to evaluate:
                  tuneGrid = gbmGrid)
print(fit.gbm2)
#tune rf model
rfGrid <- expand.grid(mtry = c(2,4,8,15))
set.seed(849)
fit.rf2 <-  train(call_cnt ~ ., data = data_train, method = 'rf',
                  trainControl = cctrlcv5, 
                  tuneGrid = rfGrid, preProcess = c('center', 'scale'))


print(fit.rf2)


#---------------------------------------------------------------------------------------------------
#Step 8. 
#Model Evaluation
#Finalize Model and estimate error
#Use the best algorithm(s) to predict on validation dataset and estimate error on unseen data.
#---------------------------------------------------------------------------------------------------

fit.glmnettrans$finalModel
fit.glmnettrans$bestTune
fit.glmnettrans$results

#variable importance Grapgh
varImp(fit.lm)
plot(varImp(fit.lm))
plot(varImp(fit.glmnetlog))

#create new variable in data_test for 

#Predict Values
predicted_callcnt<- exp(predict(fit.lmtrans,data_test))
predictmodelvalues<-data.frame(actual = data_test$call_cnt, predicted=predicted_callcnt)
head(predictmodelvalues)


#Visualze the results for the Actual and predictable values
ggplot(data = predictmodelvalues,aes(x = actual, y = predicted)) + 
  geom_point(colour = "blue") + 
  geom_abline(intercept = 0, slope = 1, colour = "red") +
  geom_vline(xintercept = 23, colour = "green", linetype = "dashed") +
    ggtitle("Predicted vs. Actual")




















