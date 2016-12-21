tuneItMan <- function(dataList, modelStr, tuningGrid, targetShift, 
                      foldCount = 5, seed = 16450, printMetrics = TRUE) {
    allMetrics <- data.frame()
    for(gridIndex in 1:nrow(tuningGrid)) {
        actGrid <- tuningGrid[gridIndex,]
        aktModelList <- list(actGrid)
        names(aktModelList) <- modelStr
        aktPred <- predictCVTrain(dataList, modelList = aktModelList, 
                                  foldCount = foldCount, targetShift = targetShift)
        aktMetrics <- evaluatePrediction(aktPred$cvPreds)
        aktMetrics <- cbind(aktMetrics, actGrid)
        if(printMetrics) {
            loginfo(paste(colnames(aktMetrics), aktMetrics, sep = ' : '))
        }
        if(nrow(allMetrics) == 0) {
            allMetrics <- aktMetrics
        } else {
            allMetrics <- rbind(allMetrics, aktMetrics)
        }
    }
    return(allMetrics)
}

predictCVTrain <- function(dataList, modelList, foldCount, targetShift, seed = 16450) {
    set.seed(seed)
    train <- dataList$train
    foldArray <- createFolds(train$y, k = foldCount, list = FALSE)
    resultList <- list()
    
    # CV Predict Trainset
    allPreds <- data.frame()
    allMetrics <- data.frame()
    for(i in unique(foldArray)) {
        trainFold <- foldData(train, foldArray, i)
        
        foldModels <- trainLevelModels(modelList = modelList, trainData = trainFold$train, seed = seed)
        foldPreds <- predictLevel(models = foldModels, testData = trainFold$test, targetShift = targetShift)
        
        #actMetrics <- evaluatePrediction(foldPreds)
        #actMetrics <- cbind('fold' = i, actMetrics)
        
        if(nrow(allPreds) == 0) {
            allPreds <- foldPreds
            #allMetrics <- actMetrics
        } else {
            allPreds <- rbind(allPreds, foldPreds)
            #allMetrics <- rbind(allMetrics, actMetrics)
        }
    }
    allPreds <- arrange(allPreds, id)
    if(is.null(allPreds$y)) {
        cvPreds <- list('id' = allPreds$id, 'y' = NULL, 
                        'predictors' = select(allPreds, -c(id)))
    } else {
        cvPreds <- list('id' = allPreds$id, 'y' = allPreds$y, 
                       'predictors' = select(allPreds, -c(id, y)))
    }
    
    # Fit models with all data
    allModels <- trainLevelModels(modelList = modelList, trainData = train, seed = seed)
    if(!is.null(dataList$test)) {
        testPreds <- predictLevel(models = allModels, testData = dataList$test, targetShift = targetShift)
    }
    
    result <- list('cvPreds' = cvPreds, 'testPreds' = testPreds)
    
    return(result)
}

trainLevelModels <- function(modelList, trainData, seed) {
    trainedModels <- list()
    for(actModelStr in names(modelList)) {
        actTrainParas <- modelList[[actModelStr]]
        #### GLMNET ####
        if(grepl('^elasticnet', actModelStr)) {
            set.seed(seed)
            actFit <- glmnet(x = trainData$predictors, y = trainData$y,
                             family = as.character(actTrainParas$family), 
                             alpha = as.numeric(actTrainParas$alpha), 
                             lambda = as.numeric(actTrainParas$lambda))
        } else if(grepl('^lasso', actModelStr)) {
            set.seed(seed)
            actFit <- glmnet(x = trainData$predictors, y = trainData$y,
                             family = as.character(actTrainParas$family), 
                             alpha = 1, 
                             lambda = as.numeric(actTrainParas$lambda))
        } else if(grepl('^ridge', actModelStr)) {
            set.seed(seed)
            actFit <- glmnet(x = trainData$predictors, y = trainData$y,
                             family = as.character(actTrainParas$family), 
                             alpha = 0, 
                             lambda = as.numeric(actTrainParas$lambda))
        } 
        #### Random Forest ####
        else if(grepl('^rf', actModelStr)) {
            set.seed(seed)
            actFit <- randomForest(x = trainData$predictors, y = trainData$y, 
                                   mtry = as.numeric(actTrainParas$mtry), 
                                   ntree = as.numeric(actTrainParas$ntree))
        } 
        #### XGBOOST ####
        else if(grepl('^xgb', actModelStr)) {
            dtrain <- xgb.DMatrix(as.matrix(trainData$predictors), label = trainData$y)
            #actTrainParas$objective <- as.character(actTrainParas$objective)
            set.seed(seed)
            actFit <- xgb.train(params = actTrainParas, data = dtrain, 
                                nrounds = actTrainParas$nrounds, verbose = 1)
        }
        
        # Constructing list
        actModelList <- list(actFit, actTrainParas)
        names(actModelList) <- c(actModelStr, 'trainParas')
        if(length(trainedModels) == 0) {
            trainedModels <- list(actModelList)
        } else {
            trainedModels <- append(trainedModels, list(actModelList))
        }
    }
    names(trainedModels) <- names(modelList)
    return(trainedModels)
}

predictLevel <- function(models, testData, targetShift) {
    preds <- data.frame('id' = testData$id)
    if(!is.null(testData$y)) {
        preds <- cbind(preds, 'y' = (exp(testData$y) - targetShift))
    }
    
    for(actModelStr in names(models)) {
        actModel <- models[[actModelStr]][[actModelStr]]
        actParas <- models[[actModelStr]][['trainParas']]
        if(grepl('^lasso', actModelStr) |
           grepl('^ridge', actModelStr) |
           grepl('^elasticnet', actModelStr)) {
            if(is.null(actParas$lambda.min)) {
                stop('lambda.min has been set!')
            }
            p <- exp(predict(object = actModel, newx = testData$predictors, 
                             s = actParas$lambda.min, type = 'response') - targetShift)
        } else if(grepl('^rf', actModelStr)) {
            p <- exp(predict(object = actModel, newdata = testData$predictors, 
                             type = 'response') - targetShift)
        } else if(grepl('^xgb', actModelStr)) {
            dtest <- xgb.DMatrix(as.matrix(testData$predictors))
            p <- exp(predict(object = actModel, newdata = dtest)) - targetShift
        }
        preds <- cbind(preds, p)
        colnames(preds)[length(colnames(preds))] <- actModelStr
    }
    return(preds)
}

evaluatePrediction <- function(predictions) {
    modelPreds <- predictions$predictors
    allMetrics <- data.frame()
    for(actModelStr in colnames(modelPreds)) {
        actPreds <- modelPreds[,actModelStr]
        actRmsle <- rmsle(actual = predictions$y, predicted = actPreds)
        #actR2 <- R2(pred = actPreds, obs = predictions$y)
        actRmse <- rmse(actual = predictions$y, predicted = actPreds)
        actMAE <- mae(actual = predictions$y, predicted = actPreds)
        # Constructing list
        actMetrics <- data.frame('model' = actModelStr, 
                                 'rmsle' = actRmsle, 
                                 #'r2' = actR2, 
                                 'rmse' = actRmse,
                                 'mae' = actMAE)
        if(nrow(allMetrics) == 0) {
            allMetrics <- actMetrics
        } else {
            allMetrics <- rbind(allMetrics, actMetrics)
        }
    }
    return(allMetrics)
}

foldData <- function(allData, foldArray, i) {
    testPreds <- allData$predictors[foldArray == i,]
    trainPreds <- allData$predictors[foldArray != i,]
    testY <- allData$y[foldArray == i]
    trainY <- allData$y[foldArray != i]
    testId <- allData$id[foldArray == i]
    trainId <- allData$id[foldArray != i]
    
    result <- list('train' = list('id' = trainId, 'y' = trainY, 'predictors' = trainPreds),
                   'test' = list('id' = testId, 'y' = testY, 'predictors' = testPreds))
    return(result)
}

createSubmission <- function(predictions, tuningGrid, properties) {
    eval <- evaluatePrediction(predictions$cvPreds)
    
    # Update submission metrics
    subFile <- 'C:/RStudioWorkspace/AllstateClaimsSeverity/tuning/submissionMetrics.Rds'
    aktTime <- format(Sys.time(), "%Y-%m-%d_%H-%M-%S")
    submissionMetrics <- readRDS(subFile)
    actMetrics <- data.frame(eval, 'time' = aktTime, 'lbScore' = NA, 'remark' = NA)
    loginfo('Submit Metrics...')
    loginfo(paste(colnames(actMetrics), actMetrics, sep = ' : '))
    submissionMetrics <- rbind(submissionMetrics, actMetrics)
    saveRDS(submissionMetrics, subFile)
    
    # Persist Tuning Results
    tResultFile <- paste('C:/RStudioWorkspace/AllstateClaimsSeverity/tuning/', as.character(eval$model),
                         '_', aktTime, '.Rds', sep = '')
    results <- list(tuneGrid = tuningGrid, properties = properties)
    saveRDS(results, tResultFile)
    
    #Create Submission File
    submissionFile <- paste('C:/RStudioWorkspace/AllstateClaimsSeverity/submissions/', as.character(eval$model),
                            '_', aktTime, '.csv', sep = '')
    submission <- data.frame('id' = predictions$testPreds$id, 'loss' = predictions$testPreds[,2])
    write.csv(submission, file=submissionFile, row.names=FALSE)
    
    loginfo('Saved Tuning and Submission Files.')
}

submitShit <- function(modelName) {
    loginfo(paste('submitting shit for', modelName))
    subData <- read.csv(paste('submissions/', modelName, '.csv', sep = ''))
    loginfo('Please insert remark...')
    remark <- scan(what=character(), nlines=1, quote = '    ')
    remark <- paste(remark, collapse = ' ')
    
    subFile <- 'tuning/submissionMetrics.Rds'
    submissions <- readRDS(subFile)
    
    modelSplits <- strsplit(modelName, '_')[[1]]
    justModelName <- modelSplits[1]
    justTime <- paste(modelSplits[2], modelSplits[3], sep = '_')
    
    submissions[submissions$time == justTime & submissions$model == justModelName, 'remark'] <- remark
    loginfo('Please insert Leaderboard Score...')
    lbScore <- scan(what=character(), nlines=1, quote = '    ')
    lbScore <- as.numeric(lbScore)
    submissions[submissions$time == justTime & submissions$model == justModelName, 'lbScore'] <- lbScore
    saveRDS(submissions, subFile)
    loginfo(paste('Saved Submission with Score:', lbScore, 'and remark:', remark))
}