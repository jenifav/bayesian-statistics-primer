import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Example 1: Simple Bayesian Parameter Estimation
# Comparing Bayesian and Frequentist approaches to estimating a proportion

def bayesian_proportion_example():
    """
    Example: Estimating the success rate of a new treatment
    Frequentist vs Bayesian approach
    """
    
    # Simulate data: 15 successes out of 20 trials
    successes = 15
    trials = 20
    
    print("=== Bayesian vs Frequentist: Estimating a Proportion ===")
    print(f"Data: {successes} successes out of {trials} trials")
    print()
    
    # Frequentist approach
    p_hat = successes / trials
    se = np.sqrt(p_hat * (1 - p_hat) / trials)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    
    print("FREQUENTIST APPROACH:")
    print(f"Point estimate: {p_hat:.3f}")
    print(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print("Interpretation: If we repeated this procedure many times,")
    print("95% of intervals would contain the true proportion.")
    print()
    
    # Bayesian approach with different priors
    priors = [
        ("Uniform (non-informative)", 1, 1),
        ("Weakly informative", 2, 2),
        ("Optimistic prior", 3, 1),
        ("Pessimistic prior", 1, 3)
    ]
    
    print("BAYESIAN APPROACH:")
    
    # Create subplot for visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (prior_name, alpha_prior, beta_prior) in enumerate(priors):
        # Posterior parameters (Beta-Binomial conjugacy)
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + (trials - successes)
        
        # Posterior statistics
        post_mean = alpha_post / (alpha_post + beta_post)
        post_var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))
        post_std = np.sqrt(post_var)
        
        # 95% credible interval
        cred_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
        cred_upper = stats.beta.ppf(0.975, alpha_post, beta_post)
        
        print(f"\n{prior_name} prior: Beta({alpha_prior}, {beta_prior})")
        print(f"Posterior: Beta({alpha_post}, {beta_post})")
        print(f"Posterior mean: {post_mean:.3f}")
        print(f"95% Credible Interval: [{cred_lower:.3f}, {cred_upper:.3f}]")
        print("Interpretation: There's a 95% probability the true proportion")
        print("lies within this interval, given the data and prior.")
        
        # Plot prior and posterior
        x = np.linspace(0, 1, 1000)
        prior_dist = stats.beta(alpha_prior, beta_prior)
        post_dist = stats.beta(alpha_post, beta_post)
        
        axes[i].plot(x, prior_dist.pdf(x), 'b--', alpha=0.7, label='Prior')
        axes[i].plot(x, post_dist.pdf(x), 'r-', linewidth=2, label='Posterior')
        axes[i].axvline(p_hat, color='green', linestyle=':', label='Frequentist estimate')
        axes[i].axvline(post_mean, color='red', linestyle='-', alpha=0.7, label='Posterior mean')
        axes[i].fill_between(x, 0, post_dist.pdf(x), 
                           where=(x >= cred_lower) & (x <= cred_upper),
                           alpha=0.3, color='red', label='95% Credible Interval')
        
        axes[i].set_title(f'{prior_name}')
        axes[i].set_xlabel('Proportion')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/bayesian_proportion_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Example 2: Bayesian A/B Testing
def bayesian_ab_test():
    """
    Example: A/B testing with Bayesian approach
    """
    
    print("\n=== Bayesian A/B Testing Example ===")
    
    # Simulate A/B test data
    np.random.seed(42)
    
    # Group A: control
    n_a = 1000
    true_rate_a = 0.10
    conversions_a = np.random.binomial(n_a, true_rate_a)
    
    # Group B: treatment  
    n_b = 1000
    true_rate_b = 0.12
    conversions_b = np.random.binomial(n_b, true_rate_b)
    
    print(f"Group A (Control): {conversions_a} conversions out of {n_a} visitors")
    print(f"Group B (Treatment): {conversions_b} conversions out of {n_b} visitors")
    print()
    
    # Frequentist approach
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    
    # Two-proportion z-test
    pooled_p = (conversions_a + conversions_b) / (n_a + n_b)
    se_diff = np.sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b))
    z_stat = (p_b - p_a) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    print("FREQUENTIST APPROACH:")
    print(f"Conversion rate A: {p_a:.3f}")
    print(f"Conversion rate B: {p_b:.3f}")
    print(f"Difference: {p_b - p_a:.3f}")
    print(f"Z-statistic: {z_stat:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    print()
    
    # Bayesian approach
    print("BAYESIAN APPROACH:")
    
    # Use uniform priors (Beta(1,1))
    alpha_prior, beta_prior = 1, 1
    
    # Posterior parameters
    alpha_a = alpha_prior + conversions_a
    beta_a = beta_prior + (n_a - conversions_a)
    alpha_b = alpha_prior + conversions_b  
    beta_b = beta_prior + (n_b - conversions_b)
    
    print(f"Posterior A: Beta({alpha_a}, {beta_a})")
    print(f"Posterior B: Beta({alpha_b}, {beta_b})")
    
    # Monte Carlo simulation to get probability that B > A
    n_samples = 100000
    samples_a = np.random.beta(alpha_a, beta_a, n_samples)
    samples_b = np.random.beta(alpha_b, beta_b, n_samples)
    
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Calculate difference distribution
    diff_samples = samples_b - samples_a
    diff_mean = np.mean(diff_samples)
    diff_std = np.std(diff_samples)
    diff_ci = np.percentile(diff_samples, [2.5, 97.5])
    
    print(f"Probability that B > A: {prob_b_better:.3f}")
    print(f"Expected difference (B - A): {diff_mean:.4f}")
    print(f"95% Credible interval for difference: [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
    
    # Probability of meaningful improvement (e.g., >1% absolute improvement)
    prob_meaningful = np.mean(diff_samples > 0.01)
    print(f"Probability of >1% improvement: {prob_meaningful:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot posterior distributions
    x = np.linspace(0, 0.2, 1000)
    post_a = stats.beta(alpha_a, beta_a)
    post_b = stats.beta(alpha_b, beta_b)
    
    axes[0].plot(x, post_a.pdf(x), 'b-', label='Group A (Control)', linewidth=2)
    axes[0].plot(x, post_b.pdf(x), 'r-', label='Group B (Treatment)', linewidth=2)
    axes[0].axvline(p_a, color='blue', linestyle='--', alpha=0.7)
    axes[0].axvline(p_b, color='red', linestyle='--', alpha=0.7)
    axes[0].set_xlabel('Conversion Rate')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Posterior Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot difference distribution
    axes[1].hist(diff_samples, bins=50, density=True, alpha=0.7, color='green')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.7, label='No difference')
    axes[1].axvline(diff_mean, color='red', linestyle='-', label='Mean difference')
    axes[1].axvline(diff_ci[0], color='red', linestyle=':', alpha=0.7)
    axes[1].axvline(diff_ci[1], color='red', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Difference (B - A)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Difference')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot probability of B > A over time (sequential analysis)
    # Simulate how probability would evolve as data accumulates
    sample_sizes = np.arange(100, n_a + 1, 50)
    probs_over_time = []
    
    for n in sample_sizes:
        # Subsample data
        conv_a_sub = int(conversions_a * n / n_a)
        conv_b_sub = int(conversions_b * n / n_b)
        
        # Update posteriors
        alpha_a_sub = alpha_prior + conv_a_sub
        beta_a_sub = beta_prior + (n - conv_a_sub)
        alpha_b_sub = alpha_prior + conv_b_sub
        beta_b_sub = beta_prior + (n - conv_b_sub)
        
        # Calculate probability
        samples_a_sub = np.random.beta(alpha_a_sub, beta_a_sub, 10000)
        samples_b_sub = np.random.beta(alpha_b_sub, beta_b_sub, 10000)
        prob_sub = np.mean(samples_b_sub > samples_a_sub)
        probs_over_time.append(prob_sub)
    
    axes[2].plot(sample_sizes, probs_over_time, 'g-', linewidth=2)
    axes[2].axhline(0.5, color='black', linestyle='--', alpha=0.7, label='No preference')
    axes[2].axhline(0.95, color='red', linestyle=':', alpha=0.7, label='Strong evidence')
    axes[2].set_xlabel('Sample Size per Group')
    axes[2].set_ylabel('P(B > A)')
    axes[2].set_title('Evidence Evolution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/bayesian_ab_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# Example 3: Bayesian Linear Regression
def bayesian_linear_regression():
    """
    Example: Bayesian linear regression with uncertainty quantification
    """
    
    print("\n=== Bayesian Linear Regression Example ===")
    
    # Generate synthetic data
    np.random.seed(123)
    n = 50
    true_intercept = 2.0
    true_slope = 1.5
    true_sigma = 0.5
    
    x = np.random.uniform(0, 5, n)
    y = true_intercept + true_slope * x + np.random.normal(0, true_sigma, n)
    
    print(f"Generated {n} data points")
    print(f"True parameters: intercept={true_intercept}, slope={true_slope}, σ={true_sigma}")
    print()
    
    # Frequentist approach (OLS)
    X = np.column_stack([np.ones(n), x])  # Design matrix
    beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
    residuals = y - X @ beta_hat
    mse = np.sum(residuals**2) / (n - 2)
    cov_matrix = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_matrix))
    
    print("FREQUENTIST APPROACH (OLS):")
    print(f"Intercept: {beta_hat[0]:.3f} ± {se[0]:.3f}")
    print(f"Slope: {beta_hat[1]:.3f} ± {se[1]:.3f}")
    print(f"Residual standard error: {np.sqrt(mse):.3f}")
    
    # 95% confidence intervals
    t_crit = stats.t.ppf(0.975, n-2)
    ci_intercept = [beta_hat[0] - t_crit*se[0], beta_hat[0] + t_crit*se[0]]
    ci_slope = [beta_hat[1] - t_crit*se[1], beta_hat[1] + t_crit*se[1]]
    
    print(f"95% CI for intercept: [{ci_intercept[0]:.3f}, {ci_intercept[1]:.3f}]")
    print(f"95% CI for slope: [{ci_slope[0]:.3f}, {ci_slope[1]:.3f}]")
    print()
    
    # Bayesian approach with conjugate priors
    print("BAYESIAN APPROACH:")
    
    # Prior parameters (weakly informative)
    # For normal-inverse-gamma conjugate prior
    mu_0 = np.array([0, 0])  # Prior mean for coefficients
    V_0 = np.eye(2) * 100    # Prior covariance (large = uninformative)
    a_0 = 0.1                # Prior shape for precision
    b_0 = 0.1                # Prior rate for precision
    
    # Posterior parameters (analytical solution for conjugate case)
    V_n = np.linalg.inv(np.linalg.inv(V_0) + X.T @ X)
    mu_n = V_n @ (np.linalg.inv(V_0) @ mu_0 + X.T @ y)
    a_n = a_0 + n/2
    b_n = b_0 + 0.5 * (y.T @ y + mu_0.T @ np.linalg.inv(V_0) @ mu_0 - mu_n.T @ np.linalg.inv(V_n) @ mu_n)
    
    # Posterior means
    print(f"Posterior mean intercept: {mu_n[0]:.3f}")
    print(f"Posterior mean slope: {mu_n[1]:.3f}")
    print(f"Posterior mean σ: {np.sqrt(b_n/a_n):.3f}")
    
    # Monte Carlo sampling from posterior
    n_samples = 10000
    
    # Sample precision (inverse variance) from Gamma
    tau_samples = np.random.gamma(a_n, 1/b_n, n_samples)
    sigma_samples = 1/np.sqrt(tau_samples)
    
    # Sample coefficients from multivariate normal
    beta_samples = np.random.multivariate_normal(mu_n, V_n, n_samples)
    
    # Posterior statistics
    intercept_samples = beta_samples[:, 0]
    slope_samples = beta_samples[:, 1]
    
    print(f"95% Credible interval for intercept: [{np.percentile(intercept_samples, 2.5):.3f}, {np.percentile(intercept_samples, 97.5):.3f}]")
    print(f"95% Credible interval for slope: [{np.percentile(slope_samples, 2.5):.3f}, {np.percentile(slope_samples, 97.5):.3f}]")
    print(f"95% Credible interval for σ: [{np.percentile(sigma_samples, 2.5):.3f}, {np.percentile(sigma_samples, 97.5):.3f}]")
    
    # Probability statements
    prob_positive_slope = np.mean(slope_samples > 0)
    prob_large_effect = np.mean(slope_samples > 1.0)
    
    print(f"Probability that slope > 0: {prob_positive_slope:.3f}")
    print(f"Probability that slope > 1.0: {prob_large_effect:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Data and fitted lines
    x_pred = np.linspace(0, 5, 100)
    
    # Frequentist prediction
    y_pred_freq = beta_hat[0] + beta_hat[1] * x_pred
    
    # Bayesian prediction with uncertainty
    y_pred_samples = intercept_samples[:, np.newaxis] + slope_samples[:, np.newaxis] * x_pred
    y_pred_mean = np.mean(y_pred_samples, axis=0)
    y_pred_ci = np.percentile(y_pred_samples, [2.5, 97.5], axis=0)
    
    axes[0,0].scatter(x, y, alpha=0.6, color='blue', label='Data')
    axes[0,0].plot(x_pred, y_pred_freq, 'r-', label='Frequentist fit', linewidth=2)
    axes[0,0].plot(x_pred, y_pred_mean, 'g-', label='Bayesian mean', linewidth=2)
    axes[0,0].fill_between(x_pred, y_pred_ci[0], y_pred_ci[1], alpha=0.3, color='green', label='95% Credible band')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    axes[0,0].set_title('Data and Fitted Models')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Posterior distributions
    axes[0,1].hist(intercept_samples, bins=50, density=True, alpha=0.7, color='blue', label='Intercept')
    axes[0,1].axvline(true_intercept, color='red', linestyle='--', label='True value')
    axes[0,1].axvline(np.mean(intercept_samples), color='blue', linestyle='-', alpha=0.7)
    axes[0,1].set_xlabel('Intercept')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Posterior: Intercept')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].hist(slope_samples, bins=50, density=True, alpha=0.7, color='green', label='Slope')
    axes[1,0].axvline(true_slope, color='red', linestyle='--', label='True value')
    axes[1,0].axvline(np.mean(slope_samples), color='green', linestyle='-', alpha=0.7)
    axes[1,0].set_xlabel('Slope')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Posterior: Slope')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].hist(sigma_samples, bins=50, density=True, alpha=0.7, color='orange', label='σ')
    axes[1,1].axvline(true_sigma, color='red', linestyle='--', label='True value')
    axes[1,1].axvline(np.mean(sigma_samples), color='orange', linestyle='-', alpha=0.7)
    axes[1,1].set_xlabel('σ')
    axes[1,1].set_ylabel('Density')
    axes[1,1].set_title('Posterior: Error Standard Deviation')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/bayesian_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    # Run all examples
    fig1 = bayesian_proportion_example()
    fig2 = bayesian_ab_test()
    fig3 = bayesian_linear_regression()
    
    print("\n=== Summary ===")
    print("These examples demonstrate key differences between Bayesian and frequentist approaches:")
    print("1. Bayesian methods provide probability statements about parameters")
    print("2. Prior information can be naturally incorporated")
    print("3. Uncertainty is fully quantified through posterior distributions")
    print("4. Results have intuitive interpretations")
    print("5. Sequential updating is natural and principled")

