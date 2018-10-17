library(class) # KNN

source("./functions/cv.R")

cv_knn = function(X,y,k=k,V,seed=NA)
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
  test.error.knn <- c()
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
    results.knn <- knn(train=X.train,test=X.test,cl=y.train,k=k)
    
    # Calcuate the test error for this fold
    test.error.knn[i] <- sum(results.knn!=y.test)/length(results.knn)
  }
  
  # Calculate the mean error over each fold
  cv.error = sum(test.error.knn)/V
  
  # Return the results
  return(c(cv.error, test.error.knn))
}