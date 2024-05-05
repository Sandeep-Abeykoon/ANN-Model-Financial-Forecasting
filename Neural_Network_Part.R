# w1912796/20210057
#Sandeep Abeykoon

# Read the data
exchange_data <- read.csv("C:/Users/sande/Desktop/Machine Learning CW/ExchangeUSD.csv")
head(exchange_data)

# Extract the exchange rates
exchange_rates <- exchange_data$USD.EUR
head(exchange_rates)

# Number of time steps
T <- length(exchange_rates)
T

# Number of time-delayed values
delay <- 4

# Initialize input and output matrices
input_matrix <- matrix(NA, nrow = T - delay, ncol = delay)
input_matrix

output_matrix <- exchange_rates[(delay + 1):T]
output_matrix

# Create input/output matrices for each time step
for (t in 1:(T - delay)) {
  input_matrix[t, ] <- exchange_rates[(t + delay - 1):(t)]
}

# Print the shapes of input and output matrices
print(paste("Input matrix shape:", dim(input_matrix)))
print(paste("Output matrix shape:", length(output_matrix)))

# converting input_matrix and output_matrix to data frames if needed
input_df <- as.data.frame(input_matrix)
output_df <- as.data.frame(output_matrix)

input_df
output_df

# Before normalization
par(mfrow=c(2,2))
for (i in 1:4) {
  hist(input_matrix[,i], main=paste("Feature", i), xlab="Value", col="lightblue")
}


# Normalize the input matrix
normalized_input_matrix <- scale(input_matrix)

# Normalize the output vector
normalized_output_vector <- scale(output_matrix)

# After normalization
par(mfrow=c(2,2))
for (i in 1:4) {
  hist(normalized_input_matrix[,i], main=paste("Feature", i, "(Normalized)"), xlab="Value", col="lightblue")
}

# Print the normalized input matrix and output vector
print("Normalized Input Matrix:")
print(normalized_input_matrix)
print("Normalized Output Vector:")
print(normalized_output_vector)
#-------------------------------------------------------------------------------

# Load required libraries
library(neuralnet)
library(Metrics)

# Assume normalized_input_matrix and normalized_output_vector are already created

# Split the data into training and testing sets
set.seed(123) # for reproducibility
train_indices <- sample(1:nrow(normalized_input_matrix), 0.8 * nrow(normalized_input_matrix))
train_data <- normalized_input_matrix[train_indices, ]
test_data <- normalized_input_matrix[-train_indices, ]
train_output <- normalized_output_vector[train_indices]
test_output <- normalized_output_vector[-train_indices]

# Initialize a list to store results
results <- list()

# Define different MLP models
models <- list(
  model1 = neuralnet(train_output ~ ., data = train_data, hidden = c(5, 2), linear.output = FALSE),
  model2 = neuralnet(train_output ~ ., data = train_data, hidden = c(10, 5), linear.output = TRUE),
  model3 = neuralnet(train_output ~ ., data = train_data, hidden = c(8, 4), linear.output = FALSE),
  model4 = neuralnet(train_output ~ ., data = train_data, hidden = c(12, 6), linear.output = TRUE),
  model5 = neuralnet(train_output ~ ., data = train_data, hidden = c(6, 3), linear.output = FALSE),
  model6 = neuralnet(train_output ~ ., data = train_data, hidden = c(7, 3), linear.output = TRUE),
  model7 = neuralnet(train_output ~ ., data = train_data, hidden = c(9, 4), linear.output = FALSE),
  model8 = neuralnet(train_output ~ ., data = train_data, hidden = c(11, 5), linear.output = TRUE),
  model9 = neuralnet(train_output ~ ., data = train_data, hidden = c(4, 2), linear.output = FALSE),
  model10 = neuralnet(train_output ~ ., data = train_data, hidden = c(8, 5), linear.output = FALSE),
  model11 = neuralnet(train_output ~ ., data = train_data, hidden = c(10, 6), linear.output = TRUE),
  model12 = neuralnet(train_output ~ ., data = train_data, hidden = c(5, 3), linear.output = FALSE),
  model13 = neuralnet(train_output ~ ., data = train_data, hidden = c(6, 4), linear.output = FALSE)
  # Add more models with different configurations
)

# Evaluate each model
for (model_name in names(models)) {
  model <- models[[model_name]]
  predictions <- predict(model, test_data)
  
  # Calculate RMSE
  rmse_val <- sqrt(mean((predictions - test_output) ^ 2))
  
  # Calculate MAE
  mae_val <- mean(abs(predictions - test_output))
  
  # Calculate MAPE
  mape_val <- mean(abs((test_output - predictions) / test_output)) * 100
  
  # Calculate sMAPE
  smape_val <- smape(test_output, predictions)
  
  # Store results
  results[[model_name]] <- c(RMSE = rmse_val, MAE = mae_val, MAPE = mape_val, sMAPE = smape_val)
}

# Print results
for (model_name in names(results)) {
  cat("Model:", model_name, "\n")
  cat("RMSE:", results[[model_name]]["RMSE"], "\n")
  cat("MAE:", results[[model_name]]["MAE"], "\n")
  cat("MAPE:", results[[model_name]]["MAPE"], "\n")
  cat("sMAPE:", results[[model_name]]["sMAPE"], "\n\n")
}

# Loop through each model and plot
for (model_name in names(models)) {
  model <- models[[model_name]]
  plot(model, main = model_name)
}

# Define the data
model_names <- c("model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9", "model10", "model11", "model12", "model13")
RMSE <- c(0.7212461, 0.2616482, 0.7224343, 0.2846507, 0.7212473, 0.221355, 0.7227448, 0.2711931, 0.7233472, 0.7235554, 0.305102, 0.720087, 0.7195579)
MAE <- c(0.4654493, 0.2036864, 0.4698938, 0.224957, 0.4640587, 0.179394, 0.4679435, 0.2077479, 0.4662606, 0.4661602, 0.2218805, 0.4640418, 0.4600212)
MAPE <- c(144.9189, 175.5788, 185.3344, 218.1645, 143.0416, 162.3982, 182.4975, 191.3537, 164.0773, 194.2058, 162.7002, 160.2533, 167.6748)
sMAPE <- c(1.291386, 0.5966998, 1.283621, 0.6061345, 1.274153, 0.5893163, 1.254175, 0.6105109, 1.261909, 1.224017, 0.6245518, 1.234658, 1.238544)
descriptions <- c("5-2 hidden layers, nonlinear output", "10-5 hidden layers, linear output", "8-4 hidden layers, nonlinear output", "12-6 hidden layers, linear output", "6-3 hidden layers, nonlinear output", "7-3 hidden layers, linear output", "9-4 hidden layers, nonlinear output", "11-5 hidden layers, linear output", "4-2 hidden layers, nonlinear output", "8-5 hidden layers, nonlinear output", "10-6 hidden layers, linear output", "5-3 hidden layers, nonlinear output", "7-4 hidden layers, linear output")
hidden_layers <- c("5-2", "10-5", "8-4", "12-6", "6-3", "7-3", "9-4", "11-5", "4-2", "8-5", "10-6", "5-3", "7-4")

# Create a matrix
performance_matrix <- data.frame(Model = model_names, RMSE, MAE, MAPE, sMAPE, Description = descriptions, HiddenLayers = hidden_layers)

# Calculate accuracy percentage
performance_matrix$Accuracy <- 100 - performance_matrix$sMAPE

# Print the matrix
print(performance_matrix)
#---------------------------------------------------------------------------------

# Extract the 6th model
model6 <- models[["model6"]]

# Predict using the 6th model on the testing data
best_predictions <- predict(model6, test_data)

# Calculate statistical indices
best_rmse <- sqrt(mean((best_predictions - test_output) ^ 2))
best_mae <- mean(abs(best_predictions - test_output))
best_mape <- mean(abs((test_output - best_predictions) / test_output)) * 100
best_smape <- smape(test_output, best_predictions)

# Print statistical indices
cat("Best Model: model6\n")
cat("RMSE:", best_rmse, "\n")
cat("MAE:", best_mae, "\n")
cat("MAPE:", best_mape, "\n")
cat("sMAPE:", best_smape, "\n\n")

# Plot a graphical representation
plot(test_output, col = "blue", type = "l", lty = 1, xlab = "Time", ylab = "Exchange Rate", main = "Best MLP Network (Model 6): Real vs Predicted Exchange Rates")
lines(best_predictions, col = "red", type = "l", lty = 2)
legend("topright", legend = c("Real", "Predicted"), col = c("blue", "red"), lty = c(1, 2), cex = 0.8)


