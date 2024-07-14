#NUR AIN BINTI LIZAM (2127942)
#GROUP PROJECT

#Step 1: Load the dataset
#Naïve Bayes
BYOD.df <- read.csv("BYOD.csv")
View(BYOD.df)

# Load necessary libraries
library(caret)
library(e1071)  # For Naïve Bayes
library(rpart)  # For Decision Trees
library(class)  # For k-NN
library(nnet)   # For Neural Networks

#------------------------------------------------------------------------------#

# Step 2: Data Cleaning and Preprocessing
# Convert relevant columns to appropriate data types (e.g., factors for categorical data)
BYOD.df$COMF_PERSONAL <- ifelse(BYOD.df$COMF_PERSONAL == "Yes", 1, 0)
BYOD.df$COMF_SENS <- ifelse(BYOD.df$COMF_SENS == "Yes", 1, 0)
BYOD.df$SECURE_PERSONAL <- ifelse(BYOD.df$SECURE_PERSONAL=="Yes",1,0)
View(BYOD.df)

# Categorize Monthly Allowance
BYOD.df$MON_ALLOWANCE <- as.character(BYOD.df$MON_ALLOWANCE)
BYOD.df$ALLOWANCE_CAT <- factor(BYOD.df$MON_ALLOWANCE, 
                                       levels = c("Below RM200", "RM201-400", "RM401-600", "RM601-800", "RM801-900", "RM999 or above"),
                                       labels = c("Very Low", "Low", "Medium", "High", "Very High", "Extremely High"))
View(BYOD.df)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

#Step 3: Descriptive Statistical Analysis
summary(BYOD.df)
boxplot(BYOD.df$CGPA, main="CGPA Boxplot", col="lightblue")

#-------------------------------------------------------------------------------#

# Step 4: Data Splitting
set.seed(1)
train.index <- sample(c(1:dim(BYOD.df)[1]), dim(BYOD.df)[1]*0.6)
train.index
train.df <- BYOD.df[train.index, selected.var]
View(train.df)
valid.df <- BYOD.df[-train.index, selected.var]
View(valid.df)
library(e1071)
BYOD.nb <- naiveBayes(OWNERSHIP ~ ., data=train.df) 

BYOD.nb

table(train.df$OWNERSHIP)

#------------------------------------------------------------------------------#

# Step 5: Algorithm Implementation and Evaluation

#Naïve Bayes
library(e1071)
BYOD.df <- read.csv("BYOD.csv")
View(BYOD.df)
selected.var <- c(1:8,10,11)
selected.var
selected.var <- c(1:11)
selected.var

# Convert columns to factors
BYOD.df$COMF_PERSONAL <- factor(BYOD.df$COMF_PERSONAL, labels = c("yes" = 2, "maybe" = 1))
BYOD.df$COMF_SENS <- factor(BYOD.df$COMF_SENS, labels = c("yes" = 2, "maybe" = 1, "no" = 0))
BYOD.df$SECURE_PERSONAL <- factor(BYOD.df$SECURE_PERSONAL, labels = c("yes" = 2, "maybe" = 1))

# Train the Naïve Bayes model
BYOD.nb <- naiveBayes(OWNERSHIP ~ ., data = BYOD.df)

# Make predictions on the validation set
nbPred <- predict(BYOD.nb, newdata = valid.df)
nbAccuracy <- sum(nbPred == valid.df$OWNERSHIP) / nrow(valid.df)
cat("Naïve Bayes Accuracy: ", nbAccuracy, "\n") 

#-------------------------------------------------------------------------------#

# DECISION TREE 

# Ensure no missing values
train.df <- na.omit(train.df)
valid.df <- na.omit(valid.df)
# Ensure consistent factor levels across training and validation sets
valid.df$MON_ALLOWANCE <- factor(valid.df$MON_ALLOWANCE, levels = levels(train.df$MON_ALLOWANCE))

library(rpart)
# Re-build the decision tree model
treeModel <- rpart(OWNERSHIP ~ ., data = train.df, method = "class")
# Make predictions
treePred <- predict(treeModel, valid.df, type = "class")
# Calculate accuracy
treeAccuracy <- sum(treePred == valid.df$OWNERSHIP) / nrow(valid.df)
# Print Decision Tree Accuracy
cat("Decision Tree Accuracy: ", treeAccuracy, "\n")
# Optional: Visualization of Decision Tree
if (length(treeModel$frame$var) > 1) {
  plot(treeModel)
  text(treeModel, use.n = TRUE, all = TRUE, cex = 0.8)
} else {
  cat("The decision tree did not split the data. Consider adjusting parameters or inspecting the data.\n")
}

#-------------------------------------------------------------------------------#
# k-NN (USING SCALED DATA)

# Check for missing values in the training dataset
missing_train <- anyNA(train.df)
if (missing_train) {
  cat("Missing values found in train.df. Handling missing values...\n")
  train.df <- na.omit(train.df)
}
# Check for missing values in the validation datasets
missing_valid <- anyNA(valid.df)
if (missing_valid) {
  cat("Missing values found in valid.df. Handling missing values...\n")
  valid.df <- na.omit(valid.df)
}
# Ensure there are no missing values in the scaled datasets
if (anyNA(train.df.scaled) || anyNA(valid.df.scaled)) {
  stop("Missing values found in scaled datasets. Check data preprocessing.")
}

# Identify numeric columns
library(class)
numeric_cols <- sapply(train.df, is.numeric)
numeric_train <- train.df[, numeric_cols]
numeric_valid <- valid.df[, numeric_cols]
# Scale numeric columns
train.df.scaled <- as.data.frame(scale(numeric_train))
valid.df.scaled <- as.data.frame(scale(numeric_valid))
# k-NN prediction
knnPred <- knn(train = train.df.scaled, test = valid.df.scaled, cl = train.df$OWNERSHIP, k = 5)
knnAccuracy <- sum(knnPred == valid.df$OWNERSHIP) / nrow(valid.df)

cat("k-NN Accuracy: ", knnAccuracy, "\n")

#-------------------------------------------------------------------------------------------------#

# NEURAL NETWORK

# (Make sure to run Data Splitting first before run the code to avoid any error)#
# Ensure no missing values
train.df <- na.omit(train.df)
valid.df <- na.omit(valid.df)
# Convert OWNERSHIP to numeric
train.df$OWNERSHIP <- as.numeric(factor(train.df$OWNERSHIP)) - 1
valid.df$OWNERSHIP <- as.numeric(factor(valid.df$OWNERSHIP)) - 1
levels(factor(train.df$OWNERSHIP))
# One-Hot Encoding for Other Categorical Variables
train.matrix <- model.matrix(~ . - 1 + OWNERSHIP, data = train.df)
valid.matrix <- model.matrix(~ . - 1 + OWNERSHIP, data = valid.df)
# Separate the Target Variable
train.target <- train.matrix[, "OWNERSHIP"]
valid.target <- valid.matrix[, "OWNERSHIP"]
train.matrix <- train.matrix[, -which(colnames(train.matrix) == "OWNERSHIP")]
valid.matrix <- valid.matrix[, -which(colnames(valid.matrix) == "OWNERSHIP")]
# Train the Neural Network
library(nnet)
nnModel <- nnet(train.matrix, train.target, size = 5, maxit = 200)
# Predictions
nnPred <- predict(nnModel, valid.matrix, type = "raw")
nnPred <- as.factor(ifelse(nnPred > 0.5, "1", "0"))
# Calculate Accuracy
nnAccuracy <- sum(nnPred == valid.df$OWNERSHIP) / nrow(valid.df)
cat("Neural Network Accuracy: ", nnAccuracy, "\n")

#-------------------------------------------------------------------------------#

# Step 6: Compare Accuracies
cat("Naïve Bayes Accuracy: ", nbAccuracy, "\n")  #declare nbaccuracy at the part naive bayes
cat("Decision Tree Accuracy: ", treeAccuracy, "\n")
cat("k-NN Accuracy: ", knnAccuracy, "\n")
cat("Neural Network Accuracy: ", nnAccuracy, "\n")

library(rpart.plot)

# Classification tree (max depth 2)
class.tree <- rpart(OWNERSHIP ~., data=BYOD.df, 
                    control=rpart.control(maxdepth=2), method="class")

# Full-grown classification tree
class.tree <- rpart(OWNERSHIP ~., data=BYOD.df, 
                    control=rpart.control(minsplit=1), method="class")

# Plot classification tree
prp(class.tree, type=1, extra=1, split.font=1, varlen=-10)

