
train.raw <- fread('data/train.csv', stringsAsFactors = FALSE, showProgress = TRUE)
test.raw <- fread('data/test.csv', stringsAsFactors = FALSE, showProgress = TRUE)

# Load best data
subFile <- 'tuning/submissionMetrics.Rds'
submissions <- readRDS(subFile)
submissions
bestDataIndex <- which.min(submissions[, 'lbScore'])
bestData <- submissions[bestDataIndex, ]
modelName <- paste(bestData$model, bestData$time, sep = '_')
tuningData <- readRDS(paste('tuning/', modelName, '.Rds', sep = ''))
paste(names(tuningData$tuneGrid), tuningData$tuneGrid, sep = ' = ')

# Set custom objective function
xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}

logcoshobj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    grad <- tanh(preds-labels)
    hess <- 1-grad*grad
    return(list(grad = grad, hess = hess))
}

cauchyobj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    c <- 3  #the lower the "slower/smoother" the loss is. Cross-Validate.
    x <-  preds-labels
    grad <- x / (x^2/c^2+1)
    hess <- -c^2*(x^2-c^2)/(x^2+c^2)^2
    return(list(grad = grad, hess = hess))
}


fairobj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    c <- 2 #the lower the "slower/smoother" the loss is. Cross-Validate.
    x <-  preds-labels
    grad <- c*x / (abs(x)+c)
    hess <- c^2 / (abs(x)+c)^2
    return(list(grad = grad, hess = hess))
}

tuningData$tuneGrid$objective <- logcoshobj

tShift <- 200
l0Data <- preprocessL0(train.raw, test.raw, normalize = tuningData$properties$preproc.normalize, 
                       targetShift = tShift, resolveSkewness = FALSE)

modelToTrain <- list('xgb' = tuningData$tuneGrid)
cvTrainPreds <- predictCVTrain(dataList = l0Data, modelList = modelToTrain, foldCount = 10,
                               targetShift = tShift, seed = 16450)

cvEvals <- evaluatePrediction(cvTrainPreds$cvPreds)
cvEvals



head(l0Data$train$predictors[,cat83])
head(l0Data$train$predictors[,cont13])

xgbModelList <- list(xgb1 = list(
    nrounds = 270,
    eta = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    gamma = 0,
    objective = 'reg:linear',
    base_score = 7
))

cvTrainPreds <- predictCVTrain(train = l0Data$train, modelList = xgbModelList, 
                               targetShift = tShift, foldCount = 5, seed = 16450)

cvEvals <- evaluatePrediction(cvTrainPreds$cvPreds)


##### Submit Shit #####
subFile <- 'tuning/submissionMetrics.Rds'
submissions <- readRDS(subFile)
submissions
#submissions <- filter(submissions, !is.na(remark))
#saveRDS(submissions, subFile)
modelName <- 'xgb_2016-11-24_10-00-45'
tuningData <- readRDS(paste('tuning/', modelName, '.Rds', sep = ''))
paste(names(tuningData$tuneGrid), tuningData$tuneGrid, sep = ' = ')

subData <- read.csv(paste('submissions/', modelName, '.csv', sep = ''))

submitShit(modelName)
