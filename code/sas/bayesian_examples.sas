/*
Bayesian Statistics Examples in SAS
Companion code for "A Frequentist's Guide to Bayesian Statistics"

This SAS program provides equivalent functionality to the Python and R examples,
demonstrating Bayesian analysis using SAS procedures including PROC MCMC,
PROC GENMOD with Bayesian options, and PROC REG with Bayesian options.

Note: Some examples require SAS/STAT with Bayesian procedures.
*/

/* Set up options for better output */
options nodate nonumber;
ods graphics on;

/* Example 1: Simple Bayesian Parameter Estimation */
/* Comparing Bayesian and Frequentist approaches to estimating a proportion */

title "Bayesian vs Frequentist: Estimating a Proportion (SAS Version)";

/* Create data: 15 successes out of 20 trials */
data proportion_data;
    successes = 15;
    trials = 20;
    failures = trials - successes;
    p_hat = successes / trials;
    
    /* Frequentist confidence interval */
    se = sqrt(p_hat * (1 - p_hat) / trials);
    ci_lower = p_hat - 1.96 * se;
    ci_upper = p_hat + 1.96 * se;
    
    put "=== Bayesian vs Frequentist: Estimating a Proportion ===";
    put "Data: " successes "successes out of " trials "trials";
    put " ";
    put "FREQUENTIST APPROACH:";
    put "Point estimate: " p_hat 6.3;
    put "95% Confidence Interval: [" ci_lower 6.3 ", " ci_upper 6.3 "]";
    put "Interpretation: If we repeated this procedure many times,";
    put "95% of intervals would contain the true proportion.";
    put " ";
run;

/* Bayesian approach using PROC MCMC with different priors */
title2 "BAYESIAN APPROACH:";

/* Prior 1: Uniform (non-informative) Beta(1,1) */
proc mcmc data=proportion_data nbi=1000 nmc=10000 seed=123 
          plots=all outpost=posterior1;
    parms p 0.5;
    prior p ~ beta(1, 1);  /* Uniform prior */
    likelihood successes ~ binomial(trials, p);
    title3 "Uniform (non-informative) prior: Beta(1, 1)";
run;

/* Calculate posterior statistics for uniform prior */
proc means data=posterior1 mean std p2_5 p97_5;
    var p;
    title3 "Posterior Statistics - Uniform Prior";
run;

/* Prior 2: Weakly informative Beta(2,2) */
proc mcmc data=proportion_data nbi=1000 nmc=10000 seed=123 
          plots=all outpost=posterior2;
    parms p 0.5;
    prior p ~ beta(2, 2);  /* Weakly informative prior */
    likelihood successes ~ binomial(trials, p);
    title3 "Weakly informative prior: Beta(2, 2)";
run;

proc means data=posterior2 mean std p2_5 p97_5;
    var p;
    title3 "Posterior Statistics - Weakly Informative Prior";
run;

/* Prior 3: Optimistic Beta(3,1) */
proc mcmc data=proportion_data nbi=1000 nmc=10000 seed=123 
          plots=all outpost=posterior3;
    parms p 0.5;
    prior p ~ beta(3, 1);  /* Optimistic prior */
    likelihood successes ~ binomial(trials, p);
    title3 "Optimistic prior: Beta(3, 1)";
run;

proc means data=posterior3 mean std p2_5 p97_5;
    var p;
    title3 "Posterior Statistics - Optimistic Prior";
run;

/* Prior 4: Pessimistic Beta(1,3) */
proc mcmc data=proportion_data nbi=1000 nmc=10000 seed=123 
          plots=all outpost=posterior4;
    parms p 0.5;
    prior p ~ beta(1, 3);  /* Pessimistic prior */
    likelihood successes ~ binomial(trials, p);
    title3 "Pessimistic prior: Beta(1, 3)";
run;

proc means data=posterior4 mean std p2_5 p97_5;
    var p;
    title3 "Posterior Statistics - Pessimistic Prior";
run;

/* Example 2: Bayesian A/B Testing */
title "Bayesian A/B Testing Example (SAS Version)";

/* Simulate A/B test data */
data ab_test_data;
    call streaminit(42);
    
    /* Group A: control */
    n_a = 1000;
    true_rate_a = 0.10;
    conversions_a = rand('binomial', true_rate_a, n_a);
    
    /* Group B: treatment */
    n_b = 1000;
    true_rate_b = 0.12;
    conversions_b = rand('binomial', true_rate_b, n_b);
    
    /* Frequentist analysis */
    p_a = conversions_a / n_a;
    p_b = conversions_b / n_b;
    
    pooled_p = (conversions_a + conversions_b) / (n_a + n_b);
    se_diff = sqrt(pooled_p * (1 - pooled_p) * (1/n_a + 1/n_b));
    z_stat = (p_b - p_a) / se_diff;
    p_value = 2 * (1 - probnorm(abs(z_stat)));
    
    put "=== Bayesian A/B Testing Example ===";
    put "Group A (Control): " conversions_a "conversions out of " n_a "visitors";
    put "Group B (Treatment): " conversions_b "conversions out of " n_b "visitors";
    put " ";
    put "FREQUENTIST APPROACH:";
    put "Conversion rate A: " p_a 6.3;
    put "Conversion rate B: " p_b 6.3;
    put "Difference: " (p_b - p_a) 6.3;
    put "Z-statistic: " z_stat 6.3;
    put "P-value: " p_value 6.3;
    if p_value < 0.05 then put "Significant at α=0.05: Yes";
    else put "Significant at α=0.05: No";
    put " ";
    
    output;
run;

/* Bayesian approach using PROC MCMC */
title2 "BAYESIAN APPROACH:";

proc mcmc data=ab_test_data nbi=2000 nmc=20000 seed=123 
          plots=all outpost=ab_posterior;
    parms p_a 0.1 p_b 0.1;
    prior p_a ~ beta(1, 1);  /* Uniform priors */
    prior p_b ~ beta(1, 1);
    likelihood conversions_a ~ binomial(n_a, p_a);
    likelihood conversions_b ~ binomial(n_b, p_b);
    
    /* Calculate derived quantities */
    difference = p_b - p_a;
    prob_b_better = (p_b > p_a);
    meaningful_improvement = (difference > 0.01);
run;

/* Calculate Bayesian statistics */
proc means data=ab_posterior mean std p2_5 p97_5;
    var p_a p_b difference;
    title3 "Posterior Statistics";
run;

/* Calculate probabilities */
proc means data=ab_posterior mean;
    var prob_b_better meaningful_improvement;
    title3 "Probability Calculations";
run;

/* Create summary dataset for probability calculations */
data ab_summary;
    set ab_posterior;
    keep difference prob_b_better meaningful_improvement;
run;

proc sql;
    select 
        mean(prob_b_better) as Prob_B_Better format=6.3,
        mean(difference) as Expected_Difference format=8.4,
        mean(meaningful_improvement) as Prob_Meaningful format=6.3
    from ab_summary;
    title3 "Key Bayesian Results";
quit;

/* Example 3: Bayesian Linear Regression */
title "Bayesian Linear Regression Example (SAS Version)";

/* Generate synthetic data */
data regression_data;
    call streaminit(123);
    
    true_intercept = 2.0;
    true_slope = 1.5;
    true_sigma = 0.5;
    
    do i = 1 to 50;
        x = rand('uniform') * 5;
        y = true_intercept + true_slope * x + rand('normal', 0, true_sigma);
        output;
    end;
    
    drop i;
run;

/* Frequentist approach using PROC REG */
title2 "FREQUENTIST APPROACH (OLS):";

proc reg data=regression_data plots=all;
    model y = x;
    title3 "Ordinary Least Squares Regression";
run;

/* Bayesian approach using PROC MCMC */
title2 "BAYESIAN APPROACH:";

proc mcmc data=regression_data nbi=2000 nmc=20000 seed=123 
          plots=all outpost=reg_posterior;
    parms intercept 0 slope 0 sigma 1;
    
    /* Weakly informative priors */
    prior intercept ~ normal(0, sd=10);
    prior slope ~ normal(0, sd=10);
    prior sigma ~ igamma(0.1, scale=0.1);
    
    /* Likelihood */
    mu = intercept + slope * x;
    likelihood y ~ normal(mu, var=sigma*sigma);
    
    /* Derived quantities for probability statements */
    slope_positive = (slope > 0);
    slope_large = (slope > 1.0);
run;

/* Posterior statistics */
proc means data=reg_posterior mean std p2_5 p97_5;
    var intercept slope sigma;
    title3 "Posterior Statistics";
run;

/* Probability calculations */
proc means data=reg_posterior mean;
    var slope_positive slope_large;
    title3 "Probability Statements";
run;

/* Alternative: Using PROC GENMOD with Bayesian option */
title2 "Alternative: PROC GENMOD with Bayesian Analysis";

proc genmod data=regression_data;
    model y = x;
    bayes seed=123 nbi=2000 nmc=20000 
          plots=all outpost=genmod_posterior;
    title3 "Bayesian Linear Regression via PROC GENMOD";
run;

/* Compare results */
proc means data=genmod_posterior mean std p2_5 p97_5;
    var intercept x;
    title3 "PROC GENMOD Posterior Statistics";
run;

/* Create diagnostic plots and summaries */
title "Model Diagnostics and Comparisons";

/* Trace plots for convergence assessment */
proc sgplot data=reg_posterior;
    series x=iteration y=intercept;
    title2 "Trace Plot - Intercept";
run;

proc sgplot data=reg_posterior;
    series x=iteration y=slope;
    title2 "Trace Plot - Slope";
run;

proc sgplot data=reg_posterior;
    series x=iteration y=sigma;
    title2 "Trace Plot - Sigma";
run;

/* Posterior density plots */
proc sgplot data=reg_posterior;
    histogram intercept / transparency=0.5;
    density intercept;
    title2 "Posterior Distribution - Intercept";
run;

proc sgplot data=reg_posterior;
    histogram slope / transparency=0.5;
    density slope;
    title2 "Posterior Distribution - Slope";
run;

proc sgplot data=reg_posterior;
    histogram sigma / transparency=0.5;
    density sigma;
    title2 "Posterior Distribution - Sigma";
run;

/* Summary comparison table */
title "Summary of All Examples";

proc print data=_null_;
    put "=== Summary ===";
    put "These SAS examples demonstrate key differences between Bayesian and frequentist approaches:";
    put "1. Bayesian methods provide probability statements about parameters";
    put "2. Prior information can be naturally incorporated";
    put "3. Uncertainty is fully quantified through posterior distributions";
    put "4. Results have intuitive interpretations";
    put "5. Sequential updating is natural and principled";
    put " ";
    put "Key SAS Procedures for Bayesian Analysis:";
    put "- PROC MCMC: Flexible Markov Chain Monte Carlo sampling";
    put "- PROC GENMOD with BAYES statement: Bayesian generalized linear models";
    put "- PROC REG with BAYES statement: Bayesian linear regression";
    put "- PROC LOGISTIC with BAYES statement: Bayesian logistic regression";
run;

/* Macro for easy replication with different datasets */
%macro bayesian_proportion(data=, success_var=, trial_var=, alpha_prior=1, beta_prior=1);
    proc mcmc data=&data nbi=1000 nmc=10000 seed=123 plots=all;
        parms p 0.5;
        prior p ~ beta(&alpha_prior, &beta_prior);
        likelihood &success_var ~ binomial(&trial_var, p);
        title "Bayesian Proportion Analysis - Beta(&alpha_prior, &beta_prior) Prior";
    run;
%mend;

/* Macro for Bayesian A/B testing */
%macro bayesian_ab_test(data=, conv_a=, n_a=, conv_b=, n_b=);
    proc mcmc data=&data nbi=2000 nmc=20000 seed=123 plots=all;
        parms p_a 0.1 p_b 0.1;
        prior p_a ~ beta(1, 1);
        prior p_b ~ beta(1, 1);
        likelihood &conv_a ~ binomial(&n_a, p_a);
        likelihood &conv_b ~ binomial(&n_b, p_b);
        
        difference = p_b - p_a;
        prob_b_better = (p_b > p_a);
        meaningful_improvement = (difference > 0.01);
        title "Bayesian A/B Test Analysis";
    run;
%mend;

/* Macro for Bayesian linear regression */
%macro bayesian_regression(data=, y_var=, x_var=);
    proc mcmc data=&data nbi=2000 nmc=20000 seed=123 plots=all;
        parms intercept 0 slope 0 sigma 1;
        
        prior intercept ~ normal(0, sd=10);
        prior slope ~ normal(0, sd=10);
        prior sigma ~ igamma(0.1, scale=0.1);
        
        mu = intercept + slope * &x_var;
        likelihood &y_var ~ normal(mu, var=sigma*sigma);
        title "Bayesian Linear Regression Analysis";
    run;
%mend;

/* Turn off graphics */
ods graphics off;

