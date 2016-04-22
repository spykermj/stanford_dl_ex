library("ggplot2")
library("reshape2")

cost <- function(theta, x_values, y_values) {
  M <- dim(x_values)[2]
  y_hat <- theta %*% x_values
  cost <- sum((y_hat - y_values)^2) / (2.0 * M)
  return(cost)
}

gradient <- function(theta, x_values, y_values) {
  M <- dim(x_values)[2]
  y_hat <- theta %*% x_values
  gradient <- (x_values %*% t(y_hat - y_values)) / M
  return(gradient)
}

train_data = read.table("housing_training.data")
test_data = read.table("housing_test.data")

variable_count = dim(train_data)[2]

train_x = train_data[,1:variable_count-1]
train_y = train_data[,variable_count]

train_x <- cbind(i = 1.0, train_x)

test_x = test_data[,1:variable_count-1]
test_y = test_data[,variable_count]

test_x <- cbind(i = 1.0, test_x)

train_x <- t(train_x)
test_x <- t(test_x)

m <- dim(train_x)[2]
n <- dim(train_x)[1]

theta <- c(rnorm(n))

system.time(res <- optim(theta, cost, gradient, train_x, train_y, method = "BFGS"))
print(res$convergence)

theta <- res$par

actual_prices <- train_y
predicted_prices <- theta %*% train_x

train_rms <- sqrt(mean((predicted_prices - actual_prices)^2))
print(sprintf('RMS training error: %f', train_rms))

actual_prices <- test_y
predicted_prices <- theta %*% test_x

test_rms <- sqrt(mean((predicted_prices - actual_prices)^2))
print(sprintf('RMS testing error: %f', test_rms))

order_indices <- order(actual_prices)

actual_prices <- actual_prices[order_indices]
predicted_prices <- predicted_prices[order_indices]

comparison <- data.frame(id=1:length(actual_prices), actual=actual_prices, predicted=predicted_prices)
comparison <- melt(comparison, id.vars=c("id"))

p <- ggplot(comparison, aes(id, value, color=variable)) + geom_point()
p <- p + ggtitle("House Price Prediction") + ylab("House price ($1000s)") + xlab("House number")
p <- p + theme(legend.title=element_blank())
print(p)