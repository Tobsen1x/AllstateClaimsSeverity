source(file = 'C:/RStudioWorkspace/AllstateClaimsSeverity/scripts/initScripting.R', 
       echo = FALSE, encoding = 'UTF-8')
loginfo('Start Tuning xgb')

args <- commandArgs(trailingOnly = TRUE)
# Test
args <- c('xgbTuning1')
propertiesFile <- paste('C:/RStudioWorkspace/AllstateClaimsSeverity/tuning/',
                        args[1], '.properties', sep = '')
loginfo(paste('Using Properties:', propertiesFile))
properties <- read.properties(propertiesFile)

train.raw <- fread('C:/RStudioWorkspace/AllstateClaimsSeverity/data/train.csv', 
                   stringsAsFactors = FALSE, showProgress = TRUE)
test.raw <- fread('C:/RStudioWorkspace/AllstateClaimsSeverity/data/test.csv', 
                  stringsAsFactors = FALSE, showProgress = TRUE)

tShift <- as.integer(properties$preproc.targetShift)
resolveSkew <- as.logical(properties$preproc.resolveSkewness)
l0Data <- preprocessL0(train.raw, test.raw, normalize = properties$preproc.normalize,
                       targetShift = tShift, resolveSkewness = resolveSkew)
dtrain = xgb.DMatrix(as.matrix(l0Data$train$predictors), label = l0Data$train$y)
dtest = xgb.DMatrix(as.matrix(l0Data$test$predictors))

## Starting parameters
xg_eval_mae <- function (yhat, dtrain) {
    y = getinfo(dtrain, "label")
    err= mae(exp(y),exp(yhat) )
    return (list(metric = "error", value = err))
}
# Objective Functions
logcoshobj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    grad <- tanh(preds-labels)
    hess <- 1-grad*grad
    return(list(grad = grad, hess = hess))
}
xgbCvNrounds <- 5
obj <- eval(parse(text = properties$objective))
xgb_params = list(
    colsample_bytree = 0.7,
    subsample = 0.7,
    gamma = 0,
    eta = 0.1,
    objective = obj,
    max_depth = 6,
    scale_pos_weight = 1,
    min_child_weight = 1,
    base_score = 7
)



seed <- as.integer(properties$seed)
maxCvNrounds <- as.integer(properties$maxCvNrounds)
fCount <- as.integer(properties$foldCount)

#### Step 1 ####
# Fix learning rate and number of estimators for tuning tree-based parameters
set.seed(seed)
xgbCv <- xgb.cv(xgb_params,
              dtrain,
              nrounds = maxCvNrounds,
              nfold = xgbCvNrounds,
              early.stop.round = 10,
              print.every.n = 10,
              verbose = 1,
              feval = xg_eval_mae,
              maximize=FALSE)

minMae <- min(xgbCv[, test.error.mean])
minMaeIndex <- which.min(xgbCv[, test.error.mean])
loginfo(paste('Best nrounds:', minMaeIndex))

#### Step 2 ####
# Tuning Max_depth and min_child_weight 

maxDepth <- eval(parse(text = properties$maxDepth))
minChildWeight <- eval(parse(text = properties$minChildWeight))

# Because best nrounds was found with just a fraction of training data:
# (1 - 1 / maxCvNrounds), best nrounds is increased

if(maxCvNrounds == 0) {
    step1Nrounds <- minMaeIndex
} else {
    step1Nrounds <- ceiling(minMaeIndex / (1 - 1 / maxCvNrounds))
}

xgbGrid2 <- expand.grid(nrounds = step1Nrounds,
            eta = xgb_params$eta,
            max_depth = maxDepth,
            min_child_weight = minChildWeight,
            subsample = xgb_params$subsample,
            colsample_bytree = xgb_params$colsample_bytree,
            scale_pos_weight = xgb_params$scale_pos_weight,
            gamma = xgb_params$gamma,
            objective = xgb_params$objective,
            base_score = xgb_params$base_score)

tuningResult2 <- tuneItMan(dataList = l0Data, modelStr = 'xgb2', tuningGrid = xgbGrid2, 
                           foldCount = fCount, targetShift = tShift, seed = seed)
minMae <- min(tuningResult2[, 'mae'])
bestIndex2 <- which.min(tuningResult2[, 'mae'])
bestGrid2 <- tuningResult2[bestIndex2,]
# Best Tuning Grid with tuned max_depth and min_child_weight
loginfo('Best Step 2')
loginfo(paste(colnames(bestGrid2), bestGrid2, sep = ' : '))

#### Step 3 ####
# Tuning Gamma
gam <- eval(parse(text = properties$gamma))
xgbGrid3 <- expand.grid(nrounds = bestGrid2$nrounds,
                        eta = bestGrid2$eta,
                        max_depth = bestGrid2$max_depth,
                        min_child_weight = bestGrid2$min_child_weight,
                        subsample = bestGrid2$subsample,
                        colsample_bytree = bestGrid2$colsample_bytree,
                        scale_pos_weight = bestGrid2$scale_pos_weight,
                        gamma = gam,
                        objective = bestGrid2$objective,
                        base_score = bestGrid2$base_score)

tuningResult3 <- tuneItMan(dataList = l0Data, modelStr = 'xgb3', tuningGrid = xgbGrid3, 
                           foldCount = fCount, targetShift = tShift, seed = seed)
minMae <- min(tuningResult3[, 'mae'])
bestIndex3 <- which.min(tuningResult3[, 'mae'])
bestGrid3 <- tuningResult3[bestIndex3,]
# Best Tuning Grid with tuned max_depth and min_child_weight and gamma
loginfo('Best Step 3')
loginfo(paste(colnames(bestGrid3), bestGrid3, sep = ' : '))

#### Step 4 ####
# Tuning subsample and colsample_bytree
subs <- eval(parse(text = properties$subsample))
colByTree <- eval(parse(text = properties$colsampleBytree))
xgbGrid4 <- expand.grid(nrounds = bestGrid3$nrounds,
                        eta = bestGrid3$eta,
                        max_depth = bestGrid3$max_depth,
                        min_child_weight = bestGrid3$min_child_weight,
                        subsample = subs,
                        colsample_bytree = colByTree,
                        scale_pos_weight = bestGrid3$scale_pos_weight,
                        gamma = bestGrid3$gamma,
                        objective = bestGrid3$objective,
                        base_score = bestGrid3$base_score)

tuningResult4 <- tuneItMan(dataList = l0Data, modelStr = 'xgb4', tuningGrid = xgbGrid4, 
                           foldCount = fCount, targetShift = tShift, seed = seed)
minMae <- min(tuningResult4[, 'mae'])
bestIndex4 <- which.min(tuningResult4[, 'mae'])
bestGrid4 <- tuningResult4[bestIndex4,]
# Best Tuning Grid with tuned max_depth, min_child_weight, gamma, subsample, colsample_bytree
loginfo('Best Step 4')
loginfo(paste(colnames(bestGrid4), bestGrid4, sep = ' : '))

#### Step 5 #### 
# TODO Tuning Regularization Parameters

#### Step 6 #### 
# Reduce Learning Rate
tunedMaxCvNrounds <- eval(parse(text = properties$tunedMaxCvNrounds))
tunedEta <- eval(parse(text = properties$tunedEta))

xgbGrid6 <- list(
    colsample_bytree = bestGrid4$colsample_bytree,
    subsample = bestGrid4$subsample,
    gamma = bestGrid4$gamma,
    eta = tunedEta,
    objective = as.character(bestGrid4$objective),
    max_depth = bestGrid4$max_depth,
    scale_pos_weight = bestGrid4$scale_pos_weight,
    min_child_weight = bestGrid4$min_child_weight,
    base_score = bestGrid4$base_score
)

set.seed(seed)
tunedXgbCv <- xgb.cv(xgbGrid6,
                dtrain,
                nrounds = tunedMaxCvNrounds,
                nfold = xgbCvNrounds,
                early.stop.round = 10,
                print.every.n = 10,
                verbose = 1,
                feval = xg_eval_mae,
                maximize=FALSE)

minMae <- min(tunedXgbCv[, test.error.mean])
minMaeIndex <- which.min(tunedXgbCv[, test.error.mean])
loginfo(paste('Best nrounds:', minMaeIndex))

if(maxCvNrounds == 0) {
    step6Nrounds <- minMaeIndex
} else {
    step6Nrounds <- ceiling(minMaeIndex / (1 - 1 / maxCvNrounds))
}

bestGrid6 <- append(xgbGrid6, list(nrounds = step6Nrounds))
# Best Tuning Grid with tuned max_depth, min_child_weight, gamma, subsample, colsample_bytree
loginfo('Best Step 6')
loginfo(paste(names(bestGrid6), bestGrid6, sep = ' : '))

#### Create Submission ####
bestXgb <- bestGrid6
xgbBestGrid <- list(
    nrounds = bestXgb$nrounds,
    colsample_bytree = bestXgb$colsample_bytree,
    subsample = bestXgb$subsample,
    gamma = bestXgb$gamma,
    eta = bestXgb$eta,
    objective = as.character(bestXgb$objective),
    max_depth = bestXgb$max_depth,
    scale_pos_weight = bestXgb$scale_pos_weight,
    min_child_weight = bestXgb$min_child_weight,
    base_score = bestXgb$base_score
)

modelList <- list('xgb' = xgbBestGrid)
predCv <- predictCVTrain(dataList = l0Data, modelList = modelList , foldCount = fCount,
                         targetShift = tShift, seed = seed)

createSubmission(predCv, xgbBestGrid, properties)

loginfo('Finished Tuning jo!')