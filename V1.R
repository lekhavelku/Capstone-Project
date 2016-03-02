#---------------------------------------------------------------------------------------
#Step 1. Prepare Data Load libraries, load dataset, check the structore of the data
#-----------------------------------------------------------------------------------
#install.packages("corrplot")
#install.packages("ellipse")
#install.packages("car")
#install.packages("caret")
#install.packages("mlbench")
#install.packages("glmnet")
#install.packages("kernlab")  --for model svm
#install.packages("rpart")
#install.packages("gbm")
#install.packages("class")
#install.packages("randomForest")
#install.packages("cubist")
install.packages("MASS")
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
# initialize the packages)
library(e1071)
library(caret) # for box-cox transformation
library(lmtest) # bptest for testing heteroscedasticity
library(gvlma) # global validation of linear model assumptions


call_cnts <- read.delim("final_dataset.csv", quote="", stringsAsFactors=TRUE)

str(call_cnts)
table(is.na(call_cnts))

colSums(is.na(call_cnts))

table(is.na(call_cnts$Temp))

call_cnts$Temp[is.na(call_cnts$Temp)] <- median(call_cnts$Temp, na.rm = TRUE)


#-----------------------------------------------
#Step 2. convert the data type. 
#Data Cleaning (remove duplicates, remove irrelevant attributes, impute/delete missing data)
#Feature Selection (remove redundant features)
#Data Transforms (transform data, if necessary, to get balanced distributions)
#----------------------------------------------------
#Correlation Plots

cor_cnt <- cor(data_train[sapply(data_train, is.numeric)])
corrplot(cor_cnt, method = "circle", order = "alphabet")
corrplot(cor_cnt, method = "square")
corrplot(cor_cnt, method = "number")
corrplot(cor_cnt, method = "shade")
corrplot(cor_cnt, method = "color")
corrplot(cor_cnt, method = "pie")
corrplot(cor_cnt, type = "upper")
corrplot(cor_cnt, order = "alphabet")

call_cnts$call_date <- as.Date(call_cnts$call_date)
call_cnts$calendar_quarter <- factor(call_cnts$calendar_quarter)
#call_cnts$calendar_month_number <- factor(call_cnts$calendar_month_number)
#call_cnts$week_day_number <- factor(call_cnts$week_day_number)
call_cnts$Working_day <- factor(call_cnts$Working_day)
call_cnts$is_a_weekend_day <- factor(call_cnts$is_a_weekend_day)
call_cnts$is_a_public_holiday <- factor(call_cnts$is_a_public_holiday)
call_cnts$season <- factor(call_cnts$season, labels = c("Winter", "Spring","Summer", "Autumn"))
call_cnts$wevents <- factor(call_cnts$wevents, labels = c("Normal", "Rain","Fog", "Snow", "Thunderstorm", "Rain-Snow","Fog_Rain", "Fog_R_Snow"))

call_cnts$calendar_month_number <- NULL
call_cnts$week_day_number   <- NULL
call_cnts$date_key  <- NULL
call_cnts$weather_Events  <- NULL

#-------------------------------------------------------------------
#Step 3. Perform data-split to get training /validation dataset
#split the data 80% train/20% test
#------------------------------------------------------------------


sample_idx <- sample(nrow(call_cnts), nrow(call_cnts)*0.8)
data_train <- call_cnts[sample_idx, ]
data_test <- call_cnts[-sample_idx, ]

str(data_train)
str(data_test)

#-------------------------------------------------------------------
#Step 4. Summarize Data
#Descriptive statistics (summary, class distributions, standard deviations, skew)
#------------------------------------------------------------------

summary(data_train)

fivenum(data_train$call_cnt)

# Two-way tables
table( data_train$call_city,data_train$call_cnt)
prop.table(table( data_train$call_city,data_train$call_cnt),1) # Row proportions
round(prop.table(table( data_train$call_city,data_train$call_cnt),1), 2) # Round col prop to 2 digits
round(100*prop.table(table( data_train$call_city,data_train$call_cnt),1), 2) # Round col prop to 2 digits (percents)
addmargins(round(prop.table(table( data_train$call_city,data_train$call_cnt),1), 2),2) # Round col prop to 2 digits


#Descriptive Statistics
sum(data_train$call_cnt)
mean(data_train$call_cnt) # Mean of all call_cnt variables
#with(data_train, mean(call_cnt))
median(data_train$call_cnt)
var(data_train$call_cnt) # Variance
sd(data_train$call_cnt) # Standard deviation
max(data_train$call_cnt) # Max value
min(data_train$call_cnt) # Min value
range(data_train$call_cnt) # Range
quantile(data_train$call_cnt) # Quantiles 25%
quantile(data_train$call_cnt, c(.3,.6,.9)) # Customized quantiles
fivenum(data_train$call_cnt) # Boxplot elements.
length(data_train$call_cnt) # Num of observations when a variable is specify
length(data_train) # Number of variables when a dataset is specify
which.max(data_train$call_cnt) # Determines the location 
which.min(data_train$call_cnt) 
data_train$call_city[which.max(data_train$call_cnt)]

# Mode by frequencies
table(data_train$call_city)
max(table(data_train$call_city))

#---------------------------------------------------------------------------------------------------
#Step 5. Data visualizations (histograms, boxplots, correlation plots, scatterplot matrix ('pairs'))
#---------------------------------------------------------------------------------------------------
ggplot(data_train, aes(x= call_city, y = call_cnt)) + geom_point(size = 2.5, color="navy") + xlab("call_city") + ylab("call counts") + 

#boxplot
a <- ggplot(data = data_train, aes(x=wevents, y=call_cnt))+geom_boxplot(fill="chartreuse4")+coord_flip()
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

# Histogram
hist(data_train$call_cnt, col="green")


mpgid <- mutate(data_train, id=as.numeric(rownames(data_train)))
mpgstack <- melt(mpgid, id="id")
pp <- qplot(value, data=mpgstack) + facet_wrap(~variable, scales="free")
ggsave("mpg-histograms.pdf", pp, scale=2)

#Scatterplots
#to identify outliers and leverage points.

plot(data_train$call_cnt) # Index plot
plot(data_train$Working_day,data_train$call_cnt)
plot(data_train$Working_day, data_train$call_cnt, main="working_day/call_cnt", xlab="working day", ylab="call_cnt", col="red")

# A scatterplot matrix
scatterplotdata <- data_train[ ,c("call_cnt","call_city","call_date","Temp", "Working_day", "is_a_public_holiday", "wevents")]
plot(scatterplotdata)

# Another way of making a scatterplot matrix, with regression lines
# and histogram/boxplot/density/qqplot/none along the diagonal
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
baseline_mean <- mean(data_train$call_cnt)

#Evaluate RMSE and MAE on the testing dat
baseline_RMSE <- sqrt(mean((baseline_mean-data_train$call_cnt)^2))
baseline_MAE <- mean(abs(baseline_mean-data_test$call_cnt))
#-------------------------------------------------------------------
#set the controls
cctrlcv5 <- trainControl(method = "cv", number = 5)
cctrlcv10 <- trainControl(method = "cv", number = 10, returnResamp = "all")
cctrlrp <- trainControl(method='repeatedcv',number=10,repeats=3)
cctrlloocv <- trainControl(method = "LOOCV")
cctrlRm <- trainControl(method = "cv", number = 3, returnResamp = "all", search = "random")

# set seed
set.seed(849)

#lm
set.seed(849)
fit.lm <- train(call_cnt~., data=data_train, method="lm", metric="RMSE", 
                preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.lm)

plot(data_train$call_cnt, fit.lm$residuals, main="Residuals vs. Predictor", xlab="call cnts", ylab="Residuals", pch=19)      
--
#BOX COX transformation
--------------
  lmMod <- lm(call_cnt ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train)
names(data_train)
bptest(lmMod)
plot(lmMod)

gvlma(lmMod)
  
transcallcnt <- BoxCoxTrans(data_train$call_cnt)

print(transcallcnt)

# append the transformed variable to data_train
data_train <- cbind(data_train, callcnt_new=predict(transcallcnt, data_train$call_cnt)) 

head(data_train,5) # view the top 6 rows


lmMod_bc <- lm(callcnt_new ~ call_city+call_date+Temp+Humidity+Windspeed+Month_name+is_a_public_holiday+Working_day+wevents+week_day_name, data=data_train)
names(data_train)

bptest(lmMod_bc)
plot(lmMod_bc)

gvlma(lmMod_bc) # checking assumptions 
---------------------------------------------------------------------

# glmnet evaluation
set.seed(849)
fit.glmnet <- train(call_cnt~., data=data_train, method="glmnet", metric="RMSE", 
                    preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.glmnet)

#K_nearest Neighbor - Non- Linear Algorithm
set.seed(849)
fit.knn <- train(call_cnt~., data=data_train, method="knn", metric="RMSE", 
                 preProc=c("center", "scale"), trControl=cctrlcv10)
print(fit.knn)

#SVM Radial Model
set.seed(849)
fit.svmRadial <- train(call_cnt~., data=data_train, method="svmRadial", metric="RMSE",
                       trControl=cctrlcv10)
print(fit.svmRadial)

# SVM Linear
set.seed(849)
fit.svmLinear <- train(call_cnt~., data=data_train, method="svmLinear", metric="RMSE", 
                       trControl=cctrlcv10)
print(fit.svmLinear)

# CART Model
set.seed(849)
fit.rpart <- train(call_cnt~., data=data_train, method="rpart", metric="RMSE", 
                   trControl=cctrlcv10)
print(fit.rpart)


# GBM Model
set.seed(849)
fit.gbm <- train(call_cnt~ ., data = data_train, method = "gbm",metric="RMSE",
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


#plot all the graphs 
#plot(fit.lm) plot(fit.glmnet) plot(fit.knn) plot(fit.svmRadial) plot(fit.rpart)

#---------------------------------------------------------------------------------------------------
#Step 7. 
# Model Selection and Model tuning
#Improve Accuracy collecting results from different modles and testing with resamples
#Algorithm Tuning: Tune parameters of the best algorithm(s) identified in Step 4
#---------------------------------------------------------------------------------------------------
cvValues <- resamples(list(glmnet = fit.glmnet , CART = fit.rpart, SVM = fit.svmRadial, lm = fit.lm, knn = fit.knn ))

summary(cvValues)
dotplot(cvValues, metric = "RMSE")
bwplot(cvValues, metric = "RMSE")



ensemblevalues1 <- resamples(list(gbm = fit.gbm, rf = fit.rf ))
summary(ensemblevalues1)
bwplot(ensemblevalues1, metric = "RMSE")

ensemblevalues2 <- resamples(list(cubist = fit.cubist, treebag = fit.treebag ))
summary(ensemblevalues2)
bwplot(ensemblevalues2, metric = "RMSE")

#visualizing the results
#There are a number of lattice plot methods to display the results:
#bwplot, dotplot, parallelplot, xyplot, splom.
splom(cvValues, metric = "RMSE")
dotplot(cvValues, metric = "RMSE")
bwplot(cvValues, metric = "RMSE")
parallelplot(cvValues, metric = "RMSE")

#comparing the Models
rmseDiffs <- diff(cvValues, metric = "RMSE")
summary(rmseDiffs)

plot(rmseDiffs)

#-------------------------------------------------------
#Tuning model with tuning parameters and Tuning grid
#-------------------------------------------------------
# tuned model for svmradial
fit.svmRadial2 <- train(call_cnt ~ ., data = data_train, method = "svmRadial",metric="RMSE",
                        preProcess = c("center", "scale"), tuneLength = 10,trControl = cctrlcv10)

#summary
print(fit.svmRadial2)

#Plot 
plot(fit.svmRadial2, metric = "RMSE" , scales = list(x = list(log = 2)))


# tune glmnet
fit.glmnet2 <- train(call_cnt ~ ., data = data_train,method = "glmnet",metric="RMSE",
                        preProcess = c("center", "scale"),tuneLength = 10,trControl = cctrlcv10)

print(fit.glmnet2)



#comparing tuned model
cvValuestune <- resamples(list(glmnet = fit.glmnet2 , SVM = fit.svmRadial2 ))

summary(cvValuestune)

#visualizing the results
bwplot(cvValuestune, metric = "RMSE")


#comparing the Models
rmseDiffstune <- diff(cvValuestune, metric = "RMSE")
summary(rmseDiffstune)

plot(rmseDiffstune)


#----------------------------------------------------
#tunegrid
# Recall as C increases, the margin tends to get wider
#------------------------------------------------------
set.seed(849)

#eGrid <- expand.grid(.alpha = (1:10) * 0.1,  .lambda = (1:10) * 0.1)

lambda.seq <- exp(seq(log(1e-5), log(1e0), length.out = 20))
fit.glmnet3  <- train(call_cnt ~ ., data = data_train,method ="glmnet",preProc=c('center','scale'),
                      trControl=cctrlrp, tuneGrid = expand.grid(alpha = 1, lambda = lambda.seq))

print(fit.glmnet3)



# train the model svmradial
fit.svmradial3 <- train(call_cnt ~ ., data = data_train,
                        method = "svmRadial",
                        tuneGrid = data.frame(.C = c(1, .5, 2),
                                              .sigma = .01), 
                        trControl = cctrlrp,
                        preProc = c("center", "scale"))

print(fit.svmradial3)

#comparing tuned model
cvValuestunegrid <- resamples(list(glmnet = fit.glmnet3 , SVM = fit.svmradial3 ))

summary(cvValuestunegrid)

#visualizing the results
dotplot(cvValuestunegrid, metric = "RMSE")
bwplot(cvValuestunegrid, metric = "RMSE")


#comparing the Models
rmseDiffstunegrid <- diff(cvValuestunegrid, metric = "RMSE")
summary(rmseDiffstunegrid)

plot(rmseDiffstunegrid)

#tune gbm 
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 10)

nrow(gbmGrid)

set.seed(825)
fit.gbm3 <- train(call_cnt ~ ., data = data_train,
                  method = "gbm",
                  trControl = cctrlcv10,
                  verbose = FALSE,
                  metric = "RMSE",
                  ## Now specify the exact models 
                  ## to evaluate:
                  tuneGrid = gbmGrid)
#summary
fit.gbm3$finalModel
fit.gbm3$results
print(fit.gbm3)
a <- ggplot(fit.gbm3)


rfGrid <- expand.grid(mtry = c(2,4,8,15))

fit.rf2 <-  train(call_cnt ~ ., data = data_train, method = 'rf',
                  trainControl = cctrlcv10, 
                  tuneGrid = rfGrid, preProcess = c('center', 'scale'))


print(fit.rf2)

b <- ggplot(fit.rf2)

fit.cubist2 <- train(call_cnt ~ ., data = data_train, method =  "cubist",
                     tuneGrid = expand.grid(.committees = c(1, 10, 50, 100),
                                             .neighbors = c(0, 1, 5, 9)),
                     trControl = cctrlcv5)

print(fit.cubist2)

c <- ggplot(fit.cubist2)



#comparing tuned model
ensemblevaluesgrid <- resamples(list(gbm = fit.gbm3 , rf = fit.rf2 ))

summary(ensemblevaluesgrid)

#visualizing the results
dotplot(cvValuestunegrid, metric = "RMSE")
bwplot(cvValuestunegrid, metric = "RMSE")






#---------------------------------------------------------------------------------------------------
#Step 8. 
#Model Evaluation
#Finalize Model and estimate error
#Use the best algorithm(s) to predict on validation dataset and estimate error on unseen data.
#---------------------------------------------------------------------------------------------------

fit.glmnet$finalModel
rf$bestTune
fit.glmnet$results

#variable importance Grapgh
varImp(fit.lm)
plot(varImp(fit.lm))
plot(varImp(fit.glmnet))
plot(varImp(fit.knn))
plot(varImp(fit.rpart))
plot(varImp(fit.svmRadial))


#Predict Values

predictedVal<-predict(fit.lm,data_test)

modelvalues<-data.frame(obs = data_test$call_cnt, pred=predictedVal)

defaultSummary(modelvalues)

#rm(residualslm) <- resid(fit.lm)

#plot(data_train$call_cnt.residuals)

#make predictions on the test set.
predictmodel2 <- predict(Model2, newdata = data_test)

#What is the sum of squared errors of the model on the test set?
sse_model <- sum((predictmodel2 - data_test$call_cnt)^2 )

sse_model


#What is the total sum of squares of the model on the test set?
sst_model <- sum((data_test$call_cnt - predict_base)^2 )

#R square

r_square <- 1 - (sse_model/sst_model)



# What is the largest absolute error that we make in our test set predictions?
max(abs(predictmodel2 - data_test$call_cnt))

#In which period (Month,Year pair) 
#do we make the largest absolute error in our prediction?

data_test$date_key[which.max(abs(predictmodel2 - data_test$call_cnt))]
















