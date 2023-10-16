
# Import & Load Library Packages.

## Install the library packages (if needed).
install.packages("e1071")
install.packages("GGally")
install.packages("fastDummies")
install.packages("heatmaply")
install.packages("randomForest")
install.packages("rsq")

library(dplyr)          # data wrangling
library(ggplot2)        # graphing
library(caret)          # machine learning functions
library(MLmetrics)      # machine learning metrics
library(car)            # VIF calculation
library(lmtest)         # linear regression model testing
library(GGally)         # correlation plot
library(e1071)          # Calculates skewness
library(gridExtra)      # Provides functions for arranging multiple grid-based plots on a page.
library(fastDummies)    # Converts categorical variables into binary "dummy" variables.
library(heatmaply)      # Creates interactive heatmaps with hierarchical clustering and dendrograms.
library(randomForest)   # Implements the random forest algorithm for classification and regression.
library(rsq)            # Calculates the coefficient of determination (R-squared) for linear regression models.
  


# Import the Medical Cost Personal Data Set.

## setwd() before to check before importing.
## setwd("C:/Users/jarro/Documents/University Related Stuff/Master in Data Science/8. MDA5023 - Forecasting Analytics")

## Import the data set into a data frame using the read.csv() function and read NA as missing values.
df_in <- read.csv(file="insurance.csv", na.strings=c("", "NA"), 
                    header=TRUE)



# Basic Data Exploration.

## Print the first 6 rows of data frame.
head(df_in) 

## Display the variable's names.
names(df_in) 

## Display the list structure.
str(df_in) 

## Display the basic descriptive statistics.
summary(df_in) 

## Display the number of rows.
nrow(df_in) 

## Display the number of columns.
ncol(df_in) 


# As we can see, we got these features:
# 
# age: age of primary beneficiary
# sex: insurance contractor gender, female, male
# bmi: Body Mass Index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg/m2) using the ratio of height to weight, ideally 18.5 to 24.9
# children: number of children covered by health insurance / number of dependents
# smoker: smoking or not
# region: the beneficiary’s residential area in the US, northeast, southeast, southwest, northwest.
# charges: individual medical costs billed by health insurance

# Since we are predicting insurance costs, charges will be our target feature.


# EDA

## Histograms & Boxplots for Numerical Variables
par(mfrow=c(3,2))

hist(df_in$age, 
     main="Histogram for Age", 
     xlab="Age", 
     border="blue", 
     col="maroon",
     xlim=c(0,100),
     breaks=50)

boxplot(df_in$age, col='maroon', xlab='Age', main='Box Plot for Age')

hist(df_in$bmi, 
     main="Histogram for BMI", 
     xlab="BMI", 
     border="blue", 
     col="maroon",
     xlim=c(0,60),
     breaks=50)

boxplot(df_in$bmi, col='maroon', xlab='BMI', main='Box Plot for BMI')

hist(df_in$charges, 
     main="Histogram for Charges", 
     xlab="Charges", 
     border="blue", 
     col="maroon",
     xlim=c(0,70000),
     breaks=50)

boxplot(df_in$charges, col='maroon', xlab='Charges', main='Box Plot for Charges')

dev.off()


## Visualizing Categorical Variables

# Bar plots for categorical variables
p1 <- ggplot(df_in, aes(x = sex, fill = sex)) +
  geom_bar() +
  ggtitle("Bar Plot of Sex") +
  xlab("Sex") +
  ylab("Count")

p2 <- ggplot(df_in, aes(x = smoker, fill = smoker)) +
  geom_bar() +
  ggtitle("Bar Plot of Smoker") +
  xlab("Smoker") +
  ylab("Count")

p3 <- ggplot(df_in, aes(x = children, fill = children)) +
  geom_bar() +
  ggtitle("Bar Plot of Children") +
  xlab("Children") +
  ylab("Count")

p4 <- ggplot(df_in, aes(x = region, fill = region)) +
  geom_bar() +
  ggtitle("Bar Plot of Region") +
  xlab("Region") +
  ylab("Count")

# Arrange plots in a grid
grid.arrange(p1, p2, p3, p4, ncol = 2)

dev.off()


## Visualizing together 

#charges in different regions
ggplot(df_in, aes(x = region, y = charges, color = region)) +
  geom_boxplot() +
  labs(title = "Medical Costs by Region") +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "none")

df_in %>%
  ggplot(aes(x=age, y=charges, color=smoker)) + 
  geom_point(size=2)

df_in %>%
  ggplot(aes(x=bmi, y=charges, color=smoker)) + 
  geom_point(size=2)

df_in %>%
  ggplot(aes(x=age, y=charges, color=children)) + 
  geom_point(size=2)

df_in %>% 
  ggplot(aes(x=age, y=charges, color=sex)) + 
  geom_point(size=2)

dev.off()


# Data Pre-processing 

## (1) Missing values
### Display the number of missing (NULL/NA) values.
colSums(is.na(df_in)) 


## (2) Descriptive stat (central tendency)
summary(df_in) 


## (3) Skewness
par(mfrow=c(2,2))
skewness(df_in$age)
hist(df_in$age)

skewness(df_in$bmi)
hist(df_in$bmi)

skewness(df_in$charges)
hist(df_in$charges)

# Log transformation on the "charges" variable in the "df_in" dataset
df_in <- df_in %>% 
  mutate(charges_log = log(charges))

# Check the skewness of the new "charges_log" variable
skewness(df_in$charges_log)
hist(df_in$charges_log)
dev.off()


## (4) Outliers 
p1 <- ggplot(df_in, aes(x = "", y = age)) + 
  geom_boxplot() + 
  ggtitle("Box Plot of Age")

p2 <- ggplot(df_in, aes(x = "", y = bmi)) + 
  geom_boxplot() + 
  ggtitle("Box Plot of BMI")

p3 <- ggplot(df_in, aes(x = "", y = charges)) + 
  geom_boxplot() + 
  ggtitle("Box Plot of Charges")

p4 <- ggplot(df_in, aes(x = "", y = charges_log)) + 
  geom_boxplot() + 
  ggtitle("Box Plot of Charges (Logged)")

grid.arrange(p1, p2, p3, p4, ncol = 2)
dev.off()

# Use the boxplot.stats function to detect outliers
age_outliers <- boxplot.stats(df_in$age)$out
bmi_outliers <- boxplot.stats(df_in$bmi)$out
charges_outliers <- boxplot.stats(df_in$charges)$out
charges_log_outliers <- boxplot.stats(df_in$charges_log)$out

# Print the number of outliers detected for each variable
cat("Number of outliers in age:", length(age_outliers), "\n")
cat("Number of outliers in bmi:", length(bmi_outliers), "\n")
cat("Number of outliers in charges:", length(charges_outliers), "\n")
cat("Number of outliers in charges (logged):", length(charges_log_outliers), "\n")


## (5) Correlation

# Feature Engineering 

# Encode sex and smoker as 0/1
df_in_corr <- df_in %>%
  mutate(sex = ifelse(sex == "male", 0, 1),
         smoker = ifelse(smoker == "no", 0, 1))

# Create dummies for children and region
df_in_corr <- df_in_corr %>%
  mutate(children = as.factor(children),
         region = as.factor(region)) %>%
  dummy_cols(remove_first_dummy = TRUE)

# Remove wanted columns
df_in_corr <- df_in_corr[, !(names(df_in_corr) %in% c('children', 'region', 'charges'))]

## Check Correlation Coefficients
heatmaply_cor(x = cor(df_in_corr), xlab = "Features",
              ylab = "Features", k_col = 2, k_row = 2)

dev.off()

# Calculate VIF for all variables
vif_values <- vif(lm(charges_log ~ ., data = df_in_corr))

# Print VIF values
vif_values




# Assumption Check

df_in <- subset(df_in, select = -charges)

# (1) Linearity
df_in %>%
  ggplot(aes(x=age, y=charges_log, color=smoker)) + 
  geom_point(size=2)


# (2) Independence
# Fit a linear regression model
model <- lm(charges_log ~ ., data = df_in)
summary(model)

# Extract the residuals and predicted values from the model
residuals <- resid(model)
predicted <- fitted(model)

# Plot the residuals against the predicted values
plot(predicted, residuals, main = "Residuals vs. Predicted Values", 
     xlab = "Predicted Values", ylab = "Residuals")

# Perform the Durbin-Watson test on the residuals
dwtest(model)


# (3) Homoscedasticity
# Plot residuals against fitted values
plot(model$fitted.values, model$residuals,
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted Values")

# Add a horizontal line at zero to help identify patterns
abline(h = 0, col = "red")

library(lmtest)

# perform Breusch-Pagan test
bp_test <- bptest(model)

# print the test result
print(bp_test)


# (4) Normality
resid <- residuals(model)

# Plot a histogram of the residuals
ggplot(data.frame(resid = resid), aes(x = resid)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  ggtitle("Histogram of Residuals")

# Plot a normal probability plot of the residuals
qqPlot(resid, main = "Normal Probability Plot of Residuals")




# Data Modelling

# Split data into training and testing sets
n_train <- round(0.8 * nrow(df_in))
train_indices <- sample(1:nrow(df_in), n_train)
Data_train <- df_in[train_indices, ]
Data_test <- df_in[-train_indices, ]

# Linear Regression Model
formula_lr <- as.formula("charges_log ~ .")

model_lr <- lm(formula_lr, data = Data_train)
print(model_lr)
summary(model_lr)

# remodel by removing sex because its p-value is > 0.05
model_lr2 <- lm(charges_log ~ age + bmi + children + smoker + region, data = Data_train)
print(model_lr2)
summary(model_lr2)

#predict data on test set
prediction_lr <- predict(model_lr2, newdata = Data_test)

#calculating Root Mean Squared Error
rmse_lr <- sqrt(mean((prediction_lr - Data_test$charges_log)^2))


# Random Forest Regression
formula_rf <- charges_log ~ age + sex + bmi + children + smoker + region
model_rf <- randomForest(formula_rf, data = Data_train, ntree = 1000)
print(model_rf)
summary(model_rf)

#predict data on test set
prediction_rf <- predict(model_rf, newdata = Data_test)

#calculating Root Mean Squared Error
rmse_rf <- sqrt(mean((prediction_rf - Data_test$charges_log)^2))

dev.off()




# Data Evaluation

# Print the RMSE
cat("Linear Regression RMSE:", rmse_lr, "\n")
cat("Random Forest RMSE:", rmse_rf, "\n")

# Linear Regression Prediction vs. Real plot
Data_test$prediction_lr <- predict(model_lr2, newdata = Data_test)
ggplot(Data_test, aes(x = prediction_lr, y = charges_log)) + 
  geom_point(color = "blue", alpha = 0.7) + 
  geom_abline(color = "red") +
  ggtitle("Prediction vs. Real values")

# Random Forest Regression Prediction vs. Real plot
Data_test$prediction_rf <- predict(model_rf, newdata = Data_test)
ggplot(Data_test, aes(x = prediction_rf, y = charges_log)) + 
  geom_point(color = "blue", alpha = 0.7) + 
  geom_abline(color = "red") +
  ggtitle("Prediction vs. Real values")

# Linear Regression Residuals vs. Linear model prediction plot
Data_test$residuals_lr <- Data_test$charges_log - Data_test$prediction_lr
ggplot(data = Data_test, aes(x = prediction_lr, y = residuals_lr)) +
  geom_pointrange(aes(ymin = 0, ymax = residuals_lr), color = "blue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = 3, color = "red") +
  ggtitle("Residuals vs. Linear model prediction")

# Random Forest Regression Residuals vs. Linear model prediction plot
Data_test$residuals_rf <- Data_test$charges_log - Data_test$prediction_rf
ggplot(data = Data_test, aes(x = prediction_rf, y = residuals_rf)) +
  geom_pointrange(aes(ymin = 0, ymax = residuals_rf), color = "blue", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = 3, color = "red") +
  ggtitle("Residuals vs. Linear model prediction")

# Linear Regression Histogram of residuals
ggplot(Data_test, aes(x = residuals_lr)) + 
  geom_histogram(bins = 15, fill = "blue") +
  ggtitle("Histogram of residuals")

# Random Forest Regression Histogram of residuals
ggplot(Data_test, aes(x = residuals_rf)) + 
  geom_histogram(bins = 15, fill = "blue") +
  ggtitle("Histogram of residuals")

dev.off()


# Data Deployment
Bob <- data.frame(age = 19,
                  bmi = 27.9,
                  sex = "male",
                  children = 0,
                  smoker = "yes",
                  region = "northwest")
print(paste0("Health care charges for Bob: ", exp(round(predict(model_rf, Bob), 2))))

Lisa <- data.frame(age = 40,
                   bmi = 50,
                   sex = "female",
                   children = 2,
                   smoker = "no",
                   region = "southeast")
print(paste0("Health care charges for Lisa: ", exp(round(predict(model_rf, Lisa), 2))))

John <- data.frame(age = 30,
                   bmi = 31.2,
                   sex = "male",
                   children = 0,
                   smoker = "yes",
                   region = "northeast")
print(paste0("Health care charges for John: ", exp(round(predict(model_rf, John), 2))))

