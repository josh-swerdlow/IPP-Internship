# We are going to make a toy model MCMC that fits a 'linear' data
# set with noise with two parameters alpha and beta which are
# normally distributed

import numpy as np
import pandas as pd
import seaborn as sns
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy as sp
sns.set(style="darkgrid", palette="muted")
rndst = np.random.RandomState(0)


def init_data(func, samples=10, sigma=0, plot=False):

    # Init DataFrame
    df = pd.DataFrame(
        {'x': rndst.choice(np.arange(samples), samples, replace=False)})

    # Generate noise
    noise = sigma * np.random.normal(0, sigma, samples)

    # Calculate y with noise and put it into the DataFrame
    df['y'] = func(df['x']) + noise

    if plot:
        sns.lmplot(x='x', y='y', data=df, fit_reg=False, legend=True)
        plt.show()

    return df


def mcmc_model(df, beta=0, gamma=0, trace=False, forest=False, post=False,
            plot=False, summary=False, geweke=False, gelman_rubin=False,
            effect_n=False):

    # Inititate a basic model #
    mdl = pm.Model()
    with mdl:
        # Priors for unknown model data
        # These represent the distributions that will
        # be looked into when tuning the parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        print('Added alpha to model...')
        sigma = pm.HalfNormal('sigma', sd=5)
        print('Added sigma to model...')
        if beta:
            beta = pm.Normal('beta', mu=0, sd=10)
            print('Added beta to model...')
        if gamma:
            gamma = pm.Normal('gamma', mu=0, sd=10)
            print('Added gamma to model...')

        # This is the expected value of our function
        # so it does not have any noise.
        y_exp = alpha * (df['x']) + beta * (df['x'])**2 + gamma

        # This is the observed value of our function
        # based on what we expect the distribution of values to be
        y_obs = pm.Normal('y_obs', mu=y_exp, sd=sigma, observed=df['y'])

        # Select MCMC Algorithm #
        step = pm.step_methods.Metropolis()
        # step = pm.step_methods.HamiltonianMC()
        # step = pm.step_methods.NUTS()

        # Initialize MCMC Algorithm
        trace = pm.sample(20000, step=step)

    BURN = 1000

# Average the parameters except for the burn-in
    a = np.mean(trace['alpha', BURN:])
    if beta:
        b = np.mean(trace['beta', BURN:])
    else:
        b = 0
    if gamma:
        g = np.mean(trace['gamma', BURN:])
    else:
        g = 0

    # Create a fitted curve based off the mean of the parameters
    y_pred = (df['x']) * a + (df['x']**2) * b + np.repeat(g, df['x'].shape[0])

# Plotting routines for the model
    # Creates a trace plot
    #   Plots sample histograms and values
    if trace:
        pm.traceplot(trace)
    # Creates a forest plot
    #   Generates a forest plot of 100*(1-alpha)% credible intervals from a trace
    if forest:
        pm.forestplot(trace)
    # Creates a posterior plot
    #   Plots posterior density
    if post:
        pm.plot_posterior(trace)
    # Creates a plot of the data and the fitted curve
    if plot:
        fig = plt.figure()
        plt.plot(df['x'], y_pred, 'r*', df['x'], df['y'], 'ko', figure=fig)
        plt.show()

# Statistics to understand effectiveness of model
    # Prints a summary of trace statistics for all the variables
    # These statistics include: mean, sd, mc_error, hpd_2.5, hpd_97.5
    if summary:
        summarized_results = pm.stats.summary(trace, varnames=mdl.cont_vars)
        print("Summarized Results: \n{}\n".format(summarized_results))

# Convergence and model validation

    # Prints geweke measurements for all the variables
        # Compare the mean of the first % of series with the mean
        # of the last % of series. x is divided into a number of
        # segments for which this difference is computed.
        # If the series is converged, this score should oscillate
        # between -1 and 1.
    # If the series is converged, this score should oscillate between -1 and 1.
    if geweke:
        geweke_results = pm.diagnostics.geweke(trace)
        print("Geweke Diagnostic: \n{}\n".format(geweke_results))

    # # Plots the geweke statistics
    # if geweke_plot:
    #     for key in geweke_results.keys:
    #         plot()


    # Prints gelman rubin measurements for all the variables
        # The Gelman-Rubin diagnostic tests for lack of convergence
        # by comparing the variance between multiple chains to the
        # variance within each chain. If convergence has been achieved,
        # the between-chain and within-chain variances should be
        # identical. To be most effective in detecting evidence for
        # nonconvergence, each chain should have been initialized to
        # starting values that are dispersed relative to the target
        # distribution.
    if gelman_rubin:
        gelman_rubin_results = pm.diagnostics.gelman_rubin(trace, varnames=mdl.cont_vars)
        print("Gelman Rubin Diagnostic: \n{}\n".format(gelman_rubin_results))

    # Measures an estimate of the effective sample size of a set of traces
    # effect_n should be greater than 200
    if effect_n:
        effect_n_results = pm.diagnostics.effective_n(trace, varnames=mdl.cont_vars)
        print("Effective Sample Size: \n{}\n".format(effect_n_results))

def lstsq_model(df, beta=0, gamma=0, plot=False):
    # print(np.ones(df['x'].shape[0]))
    A = np.vstack([df['x'], np.ones(df['x'].shape[0])]).T
    y = df['y']

    m, c = np.linalg.lstsq(A, y)[0]
    print(m, c)

    if plot:
        fig1 = plt.figure(1)
        plt.plot(df['x'], df['y'], 'o', label='Original data', markersize=10, figure=fig1)
        plt.plot(df['x'], m * df['x'] + c, 'r', label='Fitted line', figure=fig1)
        plt.legend()
        plt.show()


def poly_fit(f, df, plot=False):

    popt, pcov = sp.optimize.curve_fit(f, df['x'], df['y'])

    if plot:
        fig = plt.figure()
        plt.plot(df['x'], df['y'], 'ko',
            label='Original data', markersize=5, figure=fig)
        plt.plot(df['x'], f(df['x'], *popt), 'ro',
            label='Fitted line', figure=fig)
        plt.legend()
        plt.show()


def main(lin=False, quad=False, cubic=False, spline=False, expon=False):

    # Initialize function to be fit #
    # def linFunc(x, m, b):
    #     return m * x + b
    def quadFunc(x, a, b, c):
        return a * x**2 + b * x + c

    # Initialize some data into a DataFrame #
    # Parameters for data
    SAMPLES = 100
    sigma = 1

    # Parameters for function
    # m = 1
    # b = 2
    a = 1
    b = 0.5
    c = 5
    params = [a, b, c]

    df = init_data(lambda x: quadFunc(x, *params), SAMPLES, sigma, plot=False)

    # print('Initialized data for alpha = {}, beta = {}, gamma = {}, sigma = {}'.format(alpha, beta, gamma, sigma))

    # # Linear Fitting
    # lstsq_model(df, beta, gamma, plot=True)
    # mcmc_model(df, beta, gamma, plot=True)

    # # Quadratic Fitting
    # mcmc_model(df, beta, gamma, trace=True)
    poly_fit(quadFunc, df, plot=True)

    # Cubic Fitting

    # Spline Fitting

    # Exponential Fitting


main()








