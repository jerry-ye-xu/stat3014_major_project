library(MASS) # Discriminant analysis

source("./functions/cv.R")

cv_da = function(X,y,method=c("lda","qda"),V,seed=NA)
{
  # Set the seed
  if (!is.na(seed)) {
    set.seed(seed)
  }
  
  # Set n
  n = length(y)
  
  # Split the data up into V folds
  cvSets <- cvFolds(n, V)
  
  # Loop through each fold and calculate the error for that fold
  test.error.da <- c()
  for (i in 1:V) 
  {
    # set the indices corresponding to the training and test sets
    testInds <- cvSets$subsets[which(cvSets$which==i)]
    trainInds <- (1:n)[-testInds]
    
    # Separate y and X into the training and test sets
    y.test <- y[testInds]
    X.test <- X[ testInds,]
    y.train <- y[trainInds]
    X.train <- X[trainInds,]
    
    # Do classification on ith fold
    if (method=="lda") {
      res <- lda(y~., data=X,subset=trainInds)
    }
    if (method=="qda") {
      res <- qda(y~., data=X,subset=trainInds)
    }
    results.da = predict(res, X.test)$class
    
    # Calcuate the test error for this fold
    test.error.da[i] <- sum(results.da!=y.test)
  }
  
  # Calculate the mean error over each fold
  cv.error = sum(test.error.da)/n
  
  # Return the results
  return(cv.error)
}