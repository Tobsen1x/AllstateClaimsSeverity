cat("Loading necessary packages...\n")
suppressWarnings(suppressMessages(library(MASS)))
suppressWarnings(suppressMessages(library(car)))
suppressWarnings(suppressMessages(library(e1071)))
suppressWarnings(suppressMessages(library(ggplot2)))
suppressWarnings(suppressMessages(library(vioplot)))
suppressWarnings(suppressMessages(library(corrplot)))
suppressWarnings(suppressMessages(library(xgboost)))
suppressWarnings(suppressMessages(library(Cubist)))
suppressWarnings(suppressMessages(library(gbm)))
suppressWarnings(suppressMessages(library(kernlab)))
suppressWarnings(suppressMessages(library(logging)))
suppressWarnings(suppressMessages(library(properties)))
suppressWarnings(suppressMessages(library(glmnet)))
suppressWarnings(suppressMessages(library(Metrics)))
suppressWarnings(suppressMessages(library(outliers)))
suppressWarnings(suppressMessages(library(randomForest)))
suppressWarnings(suppressMessages(library(caret)))
suppressWarnings(suppressMessages(library(tidyr)))
suppressWarnings(suppressMessages(library(Hmisc)))
suppressWarnings(suppressMessages(library(data.table)))
suppressWarnings(suppressMessages(library(plyr)))
suppressWarnings(suppressMessages(library(dplyr)))

cat("Packages loaded.\n\n")

# Logging
basicConfig()
logfilename <- 'XGBTuning.log'
addHandler(writeToFile, file = paste('C:/RStudioWorkspace/AllstateClaimsSeverity/logs/', logfilename, sep = ''))

# Loading source
source(file = 'C:/RStudioWorkspace/AllstateClaimsSeverity/modeling/preprocessL0.R', echo = FALSE, encoding = 'UTF-8')
source(file = 'C:/RStudioWorkspace/AllstateClaimsSeverity/modeling/modeling.R', echo = FALSE, encoding = 'UTF-8')