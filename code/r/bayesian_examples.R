# Bayesian Statistics Examples in R
# Companion code for "A Frequentist's Guide to Bayesian Statistics"
# 
# This R script provides equivalent functionality to the Python examples,
# demonstrating Bayesian analysis using R packages

# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(MCMCpack)
library(bayesplot)
library(rstanarm)
library(brms)

# Set theme for better plots
theme_set(theme_minimal())

# Example 1: Simple Bayesian Parameter Estimation
# Comparing Bayesian and Frequentist approaches to estimating a proportion

bayesian_proportion_example_r <- function() {
  cat("=== Bayesian vs Frequentist: Estimating a Proportion (R Version) ===\n")
  
  # Simulate data: 15 successes out of 20 trials
  successes <- 15
  trials <- 20
  
  cat("Data:", successes, "successes out of", trials, "trials\n\n")
  
  # Frequentist approach
  p_hat <- successes / trials
  se <- sqrt(p_hat * (1 - p_hat) / trials)
  ci_lower <- p_hat - 1.96 * se
  ci_upper <- p_hat + 1.96 * se
  
  cat("FREQUENTIST APPROACH:\n")
  cat("Point estimate:", round(p_hat, 3), "\n")
  cat("95% Confidence Interval: [", round(ci_lower, 3), ",", round(ci_upper, 3), "]\n")
  cat("Interpretation: If we repeated this procedure many times,\n")
  cat("95% of intervals would contain the true proportion.\n\n")
  
  # Bayesian approach with different priors
  priors <- data.frame(
    name = c("Uniform (non-informative)", "Weakly informative", "Optimistic prior", "Pessimistic prior"),
    alpha_prior = c(1, 2, 3, 1),
    beta_prior = c(1, 2, 1, 3)
  )
  
  cat("BAYESIAN APPROACH:\n")
  
  # Create plots
  plot_list <- list()
  
  for (i in 1:nrow(priors)) {
    prior_name <- priors$name[i]
    alpha_prior <- priors$alpha_prior[i]
    beta_prior <- priors$beta_prior[i]
    
    # Posterior parameters (Beta-Binomial conjugacy)
    alpha_post <- alpha_prior + successes
    beta_post <- beta_prior + (trials - successes)
    
    # Posterior statistics
    post_mean <- alpha_post / (alpha_post + beta_post)
    
    # 95% credible interval
    cred_lower <- qbeta(0.025, alpha_post, beta_post)
    cred_upper <- qbeta(0.975, alpha_post, beta_post)
    
    cat("\n", prior_name, " prior: Beta(", alpha_prior, ",", beta_prior, ")\n", sep="")
    cat("Posterior: Beta(", alpha_post, ",", beta_post, ")\n", sep="")
    cat("Posterior mean:", round(post_mean, 3), "\n")
    cat("95% Credible Interval: [", round(cred_lower, 3), ",", round(cred_upper, 3), "]\n")
    cat("Interpretation: There's a 95% probability the true proportion\n")
    cat("lies within this interval, given the data and prior.\n")
    
    # Create data for plotting
    x <- seq(0, 1, length.out = 1000)
    prior_density <- dbeta(x, alpha_prior, beta_prior)
    post_density <- dbeta(x, alpha_post, beta_post)
    
    plot_data <- data.frame(
      x = rep(x, 2),
      density = c(prior_density, post_density),
      type = rep(c("Prior", "Posterior"), each = length(x))
    )
    
    # Create credible interval data
    ci_x <- x[x >= cred_lower & x <= cred_upper]
    ci_density <- dbeta(ci_x, alpha_post, beta_post)
    ci_data <- data.frame(x = ci_x, density = ci_density)
    
    p <- ggplot(plot_data, aes(x = x, y = density, color = type, linetype = type)) +
      geom_line(size = 1) +
      geom_ribbon(data = ci_data, aes(x = x, ymin = 0, ymax = density), 
                  fill = "red", alpha = 0.3, inherit.aes = FALSE) +
      geom_vline(xintercept = p_hat, color = "green", linetype = "dotted", size = 1) +
      geom_vline(xintercept = post_mean, color = "red", alpha = 0.7, size = 1) +
      scale_color_manual(values = c("Prior" = "blue", "Posterior" = "red")) +
      scale_linetype_manual(values = c("Prior" = "dashed", "Posterior" = "solid")) +
      labs(title = prior_name,
           x = "Proportion",
           y = "Density",
           color = "Distribution",
           linetype = "Distribution") +
      theme_minimal() +
      theme(legend.position = "bottom")
    
    plot_list[[i]] <- p
  }
  
  # Combine plots
  combined_plot <- do.call(grid.arrange, c(plot_list, ncol = 2))
  
  # Save plot
  ggsave("bayesian_proportion_example_r.png", combined_plot, 
         width = 12, height = 10, dpi = 300)
  
  return(combined_plot)
}

# Example 2: Bayesian A/B Testing
bayesian_ab_test_r <- function() {
  cat("\n=== Bayesian A/B Testing Example (R Version) ===\n")
  
  # Simulate A/B test data
  set.seed(42)
  
  # Group A: control
  n_a <- 1000
  true_rate_a <- 0.10
  conversions_a <- rbinom(1, n_a, true_rate_a)
  
  # Group B: treatment  
  n_b <- 1000
  true_rate_b <- 0.12
  conversions_b <- rbinom(1, n_b, true_rate_b)
  
  cat("Group A (Control):", conversions_a, "conversions out of", n_a, "visitors\n")
  cat("Group B (Treatment):", conversions_b, "conversions out of", n_b, "visitors\n\n")
  
  # Frequentist approach
  p_a <- conversions_a / n_a
  p_b <- conversions_b / n_b
  
  # Two-proportion z-test
  pooled_p <- (conversions_a + conversions_b) / (n_a + n_b)
  se_diff <- sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
  z_stat <- (p_b - p_a) / se_diff
  p_value <- 2 * (1 - pnorm(abs(z_stat)))
  
  cat("FREQUENTIST APPROACH:\n")
  cat("Conversion rate A:", round(p_a, 3), "\n")
  cat("Conversion rate B:", round(p_b, 3), "\n")
  cat("Difference:", round(p_b - p_a, 3), "\n")
  cat("Z-statistic:", round(z_stat, 3), "\n")
  cat("P-value:", round(p_value, 3), "\n")
  cat("Significant at α=0.05:", ifelse(p_value < 0.05, "Yes", "No"), "\n\n")
  
  # Bayesian approach
  cat("BAYESIAN APPROACH:\n")
  
  # Use uniform priors (Beta(1,1))
  alpha_prior <- 1
  beta_prior <- 1
  
  # Posterior parameters
  alpha_a <- alpha_prior + conversions_a
  beta_a <- beta_prior + (n_a - conversions_a)
  alpha_b <- alpha_prior + conversions_b  
  beta_b <- beta_prior + (n_b - conversions_b)
  
  cat("Posterior A: Beta(", alpha_a, ",", beta_a, ")\n", sep="")
  cat("Posterior B: Beta(", alpha_b, ",", beta_b, ")\n", sep="")
  
  # Monte Carlo simulation to get probability that B > A
  n_samples <- 100000
  samples_a <- rbeta(n_samples, alpha_a, beta_a)
  samples_b <- rbeta(n_samples, alpha_b, beta_b)
  
  prob_b_better <- mean(samples_b > samples_a)
  
  # Calculate difference distribution
  diff_samples <- samples_b - samples_a
  diff_mean <- mean(diff_samples)
  diff_ci <- quantile(diff_samples, c(0.025, 0.975))
  
  cat("Probability that B > A:", round(prob_b_better, 3), "\n")
  cat("Expected difference (B - A):", round(diff_mean, 4), "\n")
  cat("95% Credible interval for difference: [", round(diff_ci[1], 4), ",", round(diff_ci[2], 4), "]\n")
  
  # Probability of meaningful improvement (e.g., >1% absolute improvement)
  prob_meaningful <- mean(diff_samples > 0.01)
  cat("Probability of >1% improvement:", round(prob_meaningful, 3), "\n")
  
  # Visualization
  # Plot 1: Posterior distributions
  x <- seq(0, 0.2, length.out = 1000)
  post_a_density <- dbeta(x, alpha_a, beta_a)
  post_b_density <- dbeta(x, alpha_b, beta_b)
  
  post_data <- data.frame(
    x = rep(x, 2),
    density = c(post_a_density, post_b_density),
    group = rep(c("Group A (Control)", "Group B (Treatment)"), each = length(x))
  )
  
  p1 <- ggplot(post_data, aes(x = x, y = density, color = group)) +
    geom_line(size = 1.2) +
    geom_vline(xintercept = p_a, color = "blue", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = p_b, color = "red", linetype = "dashed", alpha = 0.7) +
    scale_color_manual(values = c("Group A (Control)" = "blue", "Group B (Treatment)" = "red")) +
    labs(title = "Posterior Distributions",
         x = "Conversion Rate",
         y = "Density",
         color = "Group") +
    theme_minimal()
  
  # Plot 2: Difference distribution
  diff_data <- data.frame(difference = diff_samples)
  
  p2 <- ggplot(diff_data, aes(x = difference)) +
    geom_histogram(bins = 50, alpha = 0.7, fill = "green", color = "black") +
    geom_vline(xintercept = 0, color = "black", linetype = "dashed", alpha = 0.7) +
    geom_vline(xintercept = diff_mean, color = "red", size = 1) +
    geom_vline(xintercept = diff_ci[1], color = "red", linetype = "dotted", alpha = 0.7) +
    geom_vline(xintercept = diff_ci[2], color = "red", linetype = "dotted", alpha = 0.7) +
    labs(title = "Distribution of Difference",
         x = "Difference (B - A)",
         y = "Count") +
    theme_minimal()
  
  # Plot 3: Probability evolution (simplified version)
  sample_sizes <- seq(100, n_a, by = 50)
  probs_over_time <- numeric(length(sample_sizes))
  
  for (i in seq_along(sample_sizes)) {
    n <- sample_sizes[i]
    # Subsample data
    conv_a_sub <- round(conversions_a * n / n_a)
    conv_b_sub <- round(conversions_b * n / n_b)
    
    # Update posteriors
    alpha_a_sub <- alpha_prior + conv_a_sub
    beta_a_sub <- beta_prior + (n - conv_a_sub)
    alpha_b_sub <- alpha_prior + conv_b_sub
    beta_b_sub <- beta_prior + (n - conv_b_sub)
    
    # Calculate probability
    samples_a_sub <- rbeta(10000, alpha_a_sub, beta_a_sub)
    samples_b_sub <- rbeta(10000, alpha_b_sub, beta_b_sub)
    probs_over_time[i] <- mean(samples_b_sub > samples_a_sub)
  }
  
  evolution_data <- data.frame(
    sample_size = sample_sizes,
    probability = probs_over_time
  )
  
  p3 <- ggplot(evolution_data, aes(x = sample_size, y = probability)) +
    geom_line(color = "green", size = 1.2) +
    geom_hline(yintercept = 0.5, color = "black", linetype = "dashed", alpha = 0.7) +
    geom_hline(yintercept = 0.95, color = "red", linetype = "dotted", alpha = 0.7) +
    labs(title = "Evidence Evolution",
         x = "Sample Size per Group",
         y = "P(B > A)") +
    theme_minimal()
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, p3, ncol = 3)
  
  # Save plot
  ggsave("bayesian_ab_test_r.png", combined_plot, 
         width = 15, height = 5, dpi = 300)
  
  return(combined_plot)
}

# Example 3: Bayesian Linear Regression using rstanarm
bayesian_linear_regression_r <- function() {
  cat("\n=== Bayesian Linear Regression Example (R Version) ===\n")
  
  # Generate synthetic data
  set.seed(123)
  n <- 50
  true_intercept <- 2.0
  true_slope <- 1.5
  true_sigma <- 0.5
  
  x <- runif(n, 0, 5)
  y <- true_intercept + true_slope * x + rnorm(n, 0, true_sigma)
  
  data <- data.frame(x = x, y = y)
  
  cat("Generated", n, "data points\n")
  cat("True parameters: intercept =", true_intercept, ", slope =", true_slope, ", σ =", true_sigma, "\n\n")
  
  # Frequentist approach (OLS)
  freq_model <- lm(y ~ x, data = data)
  freq_summary <- summary(freq_model)
  
  cat("FREQUENTIST APPROACH (OLS):\n")
  cat("Intercept:", round(coef(freq_model)[1], 3), "±", round(freq_summary$coefficients[1, 2], 3), "\n")
  cat("Slope:", round(coef(freq_model)[2], 3), "±", round(freq_summary$coefficients[2, 2], 3), "\n")
  cat("Residual standard error:", round(freq_summary$sigma, 3), "\n")
  
  # 95% confidence intervals
  ci <- confint(freq_model)
  cat("95% CI for intercept: [", round(ci[1,1], 3), ",", round(ci[1,2], 3), "]\n")
  cat("95% CI for slope: [", round(ci[2,1], 3), ",", round(ci[2,2], 3), "]\n\n")
  
  # Bayesian approach using rstanarm
  cat("BAYESIAN APPROACH (using rstanarm):\n")
  
  # Fit Bayesian model with weakly informative priors
  bayes_model <- stan_glm(y ~ x, data = data, 
                         prior_intercept = normal(0, 10),
                         prior = normal(0, 10),
                         prior_aux = exponential(1),
                         chains = 4, iter = 2000, 
                         refresh = 0)  # suppress output
  
  # Extract posterior samples
  posterior_samples <- as.data.frame(bayes_model)
  
  # Posterior statistics
  intercept_samples <- posterior_samples$`(Intercept)`
  slope_samples <- posterior_samples$x
  sigma_samples <- posterior_samples$sigma
  
  cat("Posterior mean intercept:", round(mean(intercept_samples), 3), "\n")
  cat("Posterior mean slope:", round(mean(slope_samples), 3), "\n")
  cat("Posterior mean σ:", round(mean(sigma_samples), 3), "\n")
  
  # Credible intervals
  intercept_ci <- quantile(intercept_samples, c(0.025, 0.975))
  slope_ci <- quantile(slope_samples, c(0.025, 0.975))
  sigma_ci <- quantile(sigma_samples, c(0.025, 0.975))
  
  cat("95% Credible interval for intercept: [", round(intercept_ci[1], 3), ",", round(intercept_ci[2], 3), "]\n")
  cat("95% Credible interval for slope: [", round(slope_ci[1], 3), ",", round(slope_ci[2], 3), "]\n")
  cat("95% Credible interval for σ: [", round(sigma_ci[1], 3), ",", round(sigma_ci[2], 3), "]\n")
  
  # Probability statements
  prob_positive_slope <- mean(slope_samples > 0)
  prob_large_effect <- mean(slope_samples > 1.0)
  
  cat("Probability that slope > 0:", round(prob_positive_slope, 3), "\n")
  cat("Probability that slope > 1.0:", round(prob_large_effect, 3), "\n")
  
  # Visualization
  # Plot 1: Data and fitted lines
  x_pred <- seq(0, 5, length.out = 100)
  
  # Frequentist prediction
  freq_pred <- predict(freq_model, newdata = data.frame(x = x_pred))
  
  # Bayesian prediction with uncertainty
  n_posterior_samples <- min(1000, nrow(posterior_samples))
  sample_indices <- sample(nrow(posterior_samples), n_posterior_samples)
  
  y_pred_samples <- matrix(NA, n_posterior_samples, length(x_pred))
  for (i in 1:n_posterior_samples) {
    idx <- sample_indices[i]
    y_pred_samples[i, ] <- posterior_samples$`(Intercept)`[idx] + 
                          posterior_samples$x[idx] * x_pred
  }
  
  y_pred_mean <- colMeans(y_pred_samples)
  y_pred_ci <- apply(y_pred_samples, 2, quantile, c(0.025, 0.975))
  
  pred_data <- data.frame(
    x = x_pred,
    freq_pred = freq_pred,
    bayes_mean = y_pred_mean,
    bayes_lower = y_pred_ci[1, ],
    bayes_upper = y_pred_ci[2, ]
  )
  
  p1 <- ggplot() +
    geom_point(data = data, aes(x = x, y = y), alpha = 0.6, color = "blue") +
    geom_line(data = pred_data, aes(x = x, y = freq_pred), color = "red", size = 1.2) +
    geom_line(data = pred_data, aes(x = x, y = bayes_mean), color = "green", size = 1.2) +
    geom_ribbon(data = pred_data, aes(x = x, ymin = bayes_lower, ymax = bayes_upper), 
                fill = "green", alpha = 0.3) +
    labs(title = "Data and Fitted Models",
         x = "x", y = "y") +
    theme_minimal()
  
  # Plot 2-4: Posterior distributions
  p2 <- ggplot(data.frame(intercept = intercept_samples), aes(x = intercept)) +
    geom_histogram(bins = 50, alpha = 0.7, fill = "blue", color = "black") +
    geom_vline(xintercept = true_intercept, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = mean(intercept_samples), color = "blue", alpha = 0.7, size = 1) +
    labs(title = "Posterior: Intercept", x = "Intercept", y = "Count") +
    theme_minimal()
  
  p3 <- ggplot(data.frame(slope = slope_samples), aes(x = slope)) +
    geom_histogram(bins = 50, alpha = 0.7, fill = "green", color = "black") +
    geom_vline(xintercept = true_slope, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = mean(slope_samples), color = "green", alpha = 0.7, size = 1) +
    labs(title = "Posterior: Slope", x = "Slope", y = "Count") +
    theme_minimal()
  
  p4 <- ggplot(data.frame(sigma = sigma_samples), aes(x = sigma)) +
    geom_histogram(bins = 50, alpha = 0.7, fill = "orange", color = "black") +
    geom_vline(xintercept = true_sigma, color = "red", linetype = "dashed", size = 1) +
    geom_vline(xintercept = mean(sigma_samples), color = "orange", alpha = 0.7, size = 1) +
    labs(title = "Posterior: σ", x = "σ", y = "Count") +
    theme_minimal()
  
  # Combine plots
  combined_plot <- grid.arrange(p1, p2, p3, p4, ncol = 2)
  
  # Save plot
  ggsave("bayesian_regression_r.png", combined_plot, 
         width = 12, height = 10, dpi = 300)
  
  return(combined_plot)
}

# Main execution function
run_all_examples_r <- function() {
  cat("Running Bayesian Statistics Examples in R\n")
  cat("==========================================\n")
  
  # Check if required packages are installed
  required_packages <- c("ggplot2", "dplyr", "tidyr", "gridExtra", 
                        "MCMCpack", "bayesplot", "rstanarm", "brms")
  
  missing_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
  
  if(length(missing_packages) > 0) {
    cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
    install.packages(missing_packages, repos = "https://cran.r-project.org")
  }
  
  # Load libraries (suppress messages)
  suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    library(gridExtra)
    library(MCMCpack)
    library(bayesplot)
    library(rstanarm)
    library(brms)
  })
  
  # Run examples
  tryCatch({
    fig1 <- bayesian_proportion_example_r()
    fig2 <- bayesian_ab_test_r()
    fig3 <- bayesian_linear_regression_r()
    
    cat("\n=== Summary ===\n")
    cat("These R examples demonstrate key differences between Bayesian and frequentist approaches:\n")
    cat("1. Bayesian methods provide probability statements about parameters\n")
    cat("2. Prior information can be naturally incorporated\n")
    cat("3. Uncertainty is fully quantified through posterior distributions\n")
    cat("4. Results have intuitive interpretations\n")
    cat("5. Sequential updating is natural and principled\n")
    cat("\nAll plots have been saved as PNG files.\n")
    
  }, error = function(e) {
    cat("Error running examples:", e$message, "\n")
    cat("Please ensure all required packages are properly installed.\n")
  })
}

# Run all examples if this script is executed directly
if (!interactive()) {
  run_all_examples_r()
}

