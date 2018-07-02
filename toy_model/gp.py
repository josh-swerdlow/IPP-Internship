import pymc3 as pm
import numpy as np
import scipy as sp
import pandas as pd

import matplotlib.pyplot as plt
from pymc3.gp.util import plot_gp_dist

# # set the seed
# np.random.seed(1)

# # The number of data points
# n = 100

# # The inputs to the GP, they must be arranged as a column vector
# X = np.linspace(0, 10, n)[:, None]

# # Define the true covariance function and its parameters
# l_true = 1.0
# eta_true = 3.0
# cov_func = eta_true**2 * pm.gp.cov.Matern52(1, l_true)

# # A mean function that is zero everywhere
# mean_func = pm.gp.mean.Zero()

# # The latent function values are one sample from a multivariate normal
# # Note that we have to call `eval()` because PyMC3 built on top of Theano
# f_true = np.random.multivariate_normal(mean_func(X).eval(),
#                                        cov_func(X).eval() + 1e-8 * np.eye(n), 1).flatten()

# # The observed data is the latent function plus a
# # small amount of IID Gaussian noise
# # The standard deviation of the noise is `sigma`
# sigma_true = 2.0
# y = f_true + sigma_true * np.random.randn(n)

# # Plot the data and the unobserved latent function
# fig = plt.figure(figsize=(12, 5))
# ax = fig.gca()
# ax.plot(X, f_true, "dodgerblue", lw=3, label="True f")
# ax.plot(X, y, 'ok', ms=3, alpha=0.5, label="Data")
# ax.set_xlabel("X")
# ax.set_ylabel("The true f(x)")
# plt.legend()
# plt.show()

# with pm.Model() as model:
#     l = pm.Gamma("l", alpha=2, beta=1)
#     eta = pm.HalfCauchy("eta", beta=5)

#     cov = eta**2 * pm.gp.cov.Matern52(1, l)
#     gp = pm.gp.Marginal(cov_func=cov)

#     sigma = pm.HalfCauchy("sigma", beta=5)
#     y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

#     mp = pm.find_MAP()

# # collect the results into a pandas dataframe to display
# # "mp" stands for marginal posterior
# df = pd.DataFrame({"Parameter": ["l", "eta", "sigma"],
#               "Value at MAP": [float(mp["l"]), float(mp["eta"]),
#                             float(mp["sigma"])],
#               "True value": [l_true, eta_true, sigma_true]})
# print(df)

# DOT CONDITIONAL
# --------------

# new values from x=0 to x=20
X_new = np.linspace(0, 20, 600)[:, None]

# add the GP conditional to the model, given the new X values
with model:
    f_pred = gp.conditional("f_pred", X_new)

# To use the MAP values, you can just replace the trace with
# a length-1 list with `mp`
with model:
    pred_samples = pm.sample_ppc([mp], vars=[f_pred], samples=2000)

# plot the results
fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# plot the samples from the gp posterior with samples and shading
# plot_gp_dist(ax, pred_samples["f_pred"], X_new)

# # plot the data and the true latent function
# plt.plot(X, f_true, "dodgerblue", lw=3, label="True f")
# plt.plot(X, y, 'ok', ms=3, alpha=0.5, label="Observed data")

# # axis labels and title
# plt.xlabel("X")
# plt.ylim([-13, 13])
# plt.title("Posterior distribution over $f(x)$ at the observed values")
# plt.legend()
# plt.show()

###
with model:
    y_pred = gp.conditional("y_pred", X_new, pred_noise=True)
    y_samples = pm.sample_ppc([mp], vars=[y_pred], samples=2000)

fig = plt.figure(figsize=(12, 5))
ax = fig.gca()

# posterior predictive distribution
plot_gp_dist(ax, y_samples["y_pred"], X_new, plot_samples=False,
        palette="bone_r")

# overlay a scatter of one draw of random points from the
#   posterior predictive distribution
plt.plot(X_new, y_samples["y_pred"][800, :].T, "co",
        ms=2, label="Predicted data")

# plot original data and true function
plt.plot(X, y, 'ok', ms=3, alpha=1.0, label="observed data")
plt.plot(X, f_true, "dodgerblue", lw=3, label="true f")

plt.xlabel("x")
plt.ylim([-13, 13])
plt.title("posterior predictive distribution, y_*")
plt.legend()
plt.show()

# # predict
# mu, var = gp.predict(X_new, point=mp, diag=True)
# sd = np.sqrt(var)

# # draw plot
# fig = plt.figure(figsize=(12,5)); ax = fig.gca()

# # plot mean and 2sigma intervals
# plt.plot(X_new, mu, 'r', lw=2, label="mean and 2sigma region");
# plt.plot(X_new, mu + 2*sd, 'r', lw=1); plt.plot(X_new, mu - 2*sd, 'r', lw=1);
# plt.fill_between(X_new.flatten(), mu - 2*sd, mu + 2*sd, color="r", alpha=0.5)

# # plot original data and true function
# plt.plot(X, y, 'ok', ms=3, alpha=1.0, label="observed data");
# plt.plot(X, f_true, "dodgerblue", lw=3, label="true f");

# plt.xlabel("x"); plt.ylim([-13,13]);
# plt.title("predictive mean and 2sigma interval"); plt.legend();

