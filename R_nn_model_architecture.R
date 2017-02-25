

X <- read.csv("sampledata_train.csv", header=F)
X <- as.matrix(X)
y <- read.csv("sampledata_test.csv", header=F)
y <- unlist(y)
names(y) <- NULL

# plot data
df_all <- as.data.frame(cbind(X,y))
plot(df_all, col=3-y, cex=.95, pch=20)

# Init important things
num_examples  <- nrow(X)
nn_input_dim  <- 2
nn_output_dim <- 2

epsilon       <- 0.01
reg_lambda    <- 0.01

# ==============================================
# calculate_loss function
# ==============================================
calculate_loss <- function(model) {
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
  
  # forward pass
  z1 <- X %*% W1 + b1
  a1 <- tanh(z1)
  z2 <- a1 %*% W2 + b2
  # softmax
  exp_scores <- exp(z2)
  probs <- exp_scores / rowSums(exp_scores)
  
  # calculate loss
  clb <- vector(mode='numeric', length=300)
  y2 <- y + 1
  for (i in 1: nrow(probs)) {
    clb[i] <- probs[i,y2[i]]
  }
  correct_logprobs <- -log(clb)
  data_loss <- sum(correct_logprobs)
  
  return( 1 / num_examples * data_loss)
}

# ==============================================
# predict function
# ==============================================
predict <- function(model, x) {
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
  
  # forward prop
  z1 <- x %*% W1 + b1
  a1 <- tanh(z1)
  z2 <- a1 %*% W2 + b2
  exp_scores <- exp(z2)
  probs <- exp_scores / sum(exp_scores)
  preds <- apply(probs, 1, which.max)
  return( preds)
}

# ==============================================
# build model
# ==============================================
build_model <- function(nn_hdim, num_passes=20000, print_loss=FALSE) {
  set.seed(0)
  W1 = matrix(rnorm(nn_input_dim * nn_hdim), ncol=nn_hdim)
  b1 = rep(0, nn_hdim)
  W2 = matrix(rnorm(nn_hdim * nn_output_dim), nrow=nn_hdim)
  b2 = rep(0, nn_output_dim)
  
  # what we return at the end
  model = list()
  
  # Gradient descent. For each batch:
  for (i in 1:num_passes) {
    
    z1 <- X %*% W1 + b1
    a1 <- tanh(z1)
    z2 <- a1 %*% W2 + b2
    exp_scores <- exp(z2)
    probs <- exp_scores / rowSums(exp_scores)
    
    # backpropagation
    delta3 <- probs
    for (i in 1:nrow(delta3)) {
      delta3[i,y2[i]] <- delta3[i,y2[i]] - 1
    }
    dW2 <- t(a1) %*% delta3
    db2 <- colSums(delta3)
    delta2 <- delta3 %*% t(W2) * (1 - a1**2)
    dW1 <- t(X) %*% delta2
    db1 <- colSums(delta2)
    
    # Regularization
    # skip
    
    # Gradient Descent parameter update
    W1 <- W1 + (-epsilon * dW1)
    b1 <- b1 + (-epsilon * db1)
    W2 <- W2 + (-epsilon * dW2)
    b2 <- b2 + (-epsilon * db2)
    
    # assign new parameters to model
    model$W1 <- W1
    model$W2 <- W2
    model$b1 <- b1
    model$b2 <- b2
    
    
    if (as.integer(i) %% 1000 == 0) {
      sprintf("Loss after iteration %s: %s", i, calculate_loss(model))
    }
    #return( model)
  }
  return( model)
}

mod <- build_model(4, num_passes = 2500, print_loss=T)

preds <- predict(mod, X)
sprintf("Model Accuracy: %s", mean(preds-1==y))

