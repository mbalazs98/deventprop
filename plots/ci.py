import scipy.stats as stats

N_t = 2264
observed_accuracies = [0.93, 0.94, 0.95]
for acc in observed_accuracies:
    alpha_post = int((1-acc) * N_t)
    beta_post = N_t - alpha_post
    ci = stats.beta.interval(0.95, alpha_post, beta_post)
    print(f"Observed Accuracy: {acc*100:.2f}%")
    print(f"Posterior Parameters: alpha = {alpha_post}, beta = {beta_post}")
    print(f"95% Credible Interval: {ci}")
