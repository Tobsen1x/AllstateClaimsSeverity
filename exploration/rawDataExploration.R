train.raw <- read.csv(file = 'data/train.csv', stringsAsFactors = FALSE)
test.raw <- read.csv(file = 'data/test.csv', stringsAsFactors = FALSE)

str(train.raw)
colnames(train.raw)
colnames(test.raw)

train.id <- train.raw$id
test.id <- test.raw$id
train.target <- train.raw$loss
describe(train.target)
skewness(log(train.target))

allRaw <- rbind(select(train.raw, -loss), test.raw)
allRaw[sapply(allRaw, is.character)] <- lapply(allRaw[sapply(allRaw, is.character)], 
                                       as.factor)

sapply(allRaw[sapply(allRaw, is.numeric)], skewness)

describe(allRaw$cat1)
ggplot(train.raw, aes(y = loss, x = cat1)) + 
    geom_boxplot()

ggplot(train.raw, aes(y = loss, x = cont2)) +
    geom_point(shape = 1) +
    geom_smooth(method = lm, color = 'red')



### NA exploration
res <- sapply(allRaw, FUN = function(x) {
    sum(is.na(x))
})
# => NO NAs
