preprocessL0 <- function(trainInp, testInp, normalize, targetShift, resolveSkewness) {
    train.id <- trainInp$id
    test.id <- testInp$id
    train.target <- trainInp$loss
    # Transform target
    target.log <- log(trainInp[, 'loss', with = FALSE] + targetShift)[['loss']]
    
    trainInp[, c('id', 'loss') := NULL]
    testInp[, c('id') := NULL]
    
    ntrain <- nrow(trainInp)
    allObs <- rbind(trainInp, testInp)
    
    # skewness resolving config
    csboxcoxfeats <- c('cont5')
    cssrootfeats <- c('cont1', 'cont4', 'cont6', 'cont7', 'cont8', 'cont9', 'cont10', 
                      'cont11', 'cont12', 'cont13', 'cont14')
    cssquaredfeats <- c('cont2')
    cslog1feats <- c()
    csfeats <- c('cont3')
    
    features <- names(allObs)
    for (f in features) {
        # Categorical Predictors
        if (class(allObs[[f]]) == 'character') {
            if(normalize == 'ordinal') {
                levels <- unique(allObs[[f]])
                allObs[[f]] <- as.integer(factor(allObs[[f]], levels=levels))
            } else if(normalize == 'oneHot') {
                levels <- unique(allObs[[f]])
                allObs[[f]] <- factor(allObs[[f]], levels=levels)
            } 
            # Lexographical Order
            else if(normalize == 'lexOrdinal') {
                var <- allObs[[f]]
                uni <- unique(var)
                sortedUni <- sort(uni)
                newVar <- ordered(var, levels = sortedUni)
                allObs[[f]] <- as.integer(newVar)
            }
        }
        # Numeric Predictors
        else {
            if(resolveSkewness) {
                contvar <- allObs[[f]]
                ### Resolve Scewness
                if(f %in% csboxcoxfeats) {
                    boxCoxTrans <- BoxCoxTrans(y = (contvar + 1) )
                    transVar <- predict(boxCoxTrans, contvar)
                } else if(f %in% cssrootfeats) {
                    transVar <- sqrt(contvar)
                } else if(f %in% cssquaredfeats) {
                    transVar <- contvar ^ 2
                } else if(f %in% cslog1feats) {
                    transVar <- log(contvar + 1)
                } else if(f %in% csfeats) {
                    transVar <- contvar
                } else {
                    stop(paste('No Transformation Configuration for', f))
                }
                
                # Center And Scale
                centerScaleModel <- preProcess(as.data.frame(transVar), method = c('center', 'scale'))
                centerScaledTrans <- predict(centerScaleModel, as.data.frame(transVar))
                allObs[[f]] <- centerScaledTrans[['transVar']]
            }
        }
    }
    
    if(normalize == 'oneHot') {
        allObs.sparse <- sparse.model.matrix( ~ .-1, data = allObs)
        allObs <- as.matrix(allObs.sparse)
    } 

    xTrain <- allObs[1:ntrain,]
    xTest <- allObs[(ntrain+1):nrow(allObs),]
    
    l0Train <- list('id' = train.id, 'y' = target.log, 
                    'predictors' = xTrain)
    l0Test <- list('id' = test.id,
                   'predictors' = xTest)
    
    l0Data <- list('train' = l0Train, 'test' = l0Test)
    return(l0Data)
    
    #Factorize character columns - The DataFrame way
    #allRaw[sapply(allRaw, is.character)] <- lapply(allRaw[sapply(allRaw, is.character)], 
    #                                               as.factor)
    
    #skew <- sapply(allRaw[sapply(allRaw, is.numeric)], skewness)
    # TODO Tranform Cont Predictors
    
    # One-Hot Encoding - Dauert Terrorlang
    #dmyVars <- dummyVars( ~ ., data = allRaw)
    #data <- data.frame(predict(dmyVars, newdata = allRaw))
}