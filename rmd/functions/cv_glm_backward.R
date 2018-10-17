source("./functions/cv.R")

cv_glm_backward = function(X,y,V,seed=NA,pen)
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
  test.error <- c()
  for (i in 1:V) 
  {
    # set the indices corresponding to the training and test sets
    testInds <- cvSets$subsets[which(cvSets$which==i)]
    trainInds <- (1:n)[-testInds]
    
    X = data.frame(X)
    
    # Separate y and X into the training and test sets
    y.test <- y[testInds]
    X.test <- X[ testInds,]
    y.train <- y[trainInds]
    X.train <- X[trainInds,]
  
    # Do classification on ith fold
    full <- glm(y.train ~., data=X.train, family=binomial)
    res.step <- step(full,k=pen)
    res = round(predict(res.step, newdata=X.test, type="response"))
    
    # Calcuate the test error for this fold
    test.error[i] <- sum(res!=y.test)
  }
  
  # Calculate the mean error over each fold
  cv.error = sum(test.error)/n
  
  # Return the results
  return(cv.error)
}