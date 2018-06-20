# Python file containing our different models:
#   1. Markov Chain Monte Carlo
#   2. Gaussian Process
#   3. Least Square
#   4. Combination of 1 and 2

## Note these should all probably be subclasses of initData?? ##


class markovChainMonteCarlo():
    """ Initiates a MCMC regression algorithm for the given data
    """

    def __init__(self, data, )

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
        self.trace = pm.sample(20000, step=step)

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

    def returnTrace(self):
        """Returns the trace of the MCMC Algorithm"""
        pass
    def returnMeanParams(self):
        """Returns the average of the parameters except for the burn in"""

# Plotting Routines
    def pltTrace(self):
        """Plots sample histograms and values"""
        pm.traceplot(trace)

    def pltForest(self):
        """Generates a forest plot of 100*(1-alpha)% credible intervals from a trace"""
        pm.forestplot(trace)

    def pltPosterior(self):
        """Plots posterior density"""
        pm.plot_posterior(trace)

    def pltFit(self):
        """Creates a plot of the data and the fitted curve"""
        fig = plt.figure()
        plt.plot(df['x'], y_pred, 'r*', df['x'], df['y'], 'ko', figure=fig)
        plt.show()

# Statistics to understand effectiveness of model
    def statSummary(self):
        """ Prints a summary of trace statistics for all the variables
        These statistics include: mean, sd, mc_error, hpd_2.5, hpd_97.5
        """
        summarized_results = pm.stats.summary(trace, varnames=mdl.cont_vars)
        print("Summarized Results: \n{}\n".format(summarized_results))

# Convergence and model validation
    def geweke(self):
        """Prints geweke measurements for all the variables

        Compare the mean of the first % of series with the mean of the
        last % of series. x is divided into a number of segments for
        which this difference is computed. If the series is converged,
        this score should oscillate between -1 and 1.
        """
        geweke_results = pm.diagnostics.geweke(trace)
        print("Geweke Diagnostic: \n{}\n".format(geweke_results))

    # # Plots the geweke statistics
    # if geweke_plot:
    #     for key in geweke_results.keys:
    #         plot()

    def gelmanRubin(self):
        """Prints gelman rubin measurements for all the variables

        The Gelman-Rubin diagnostic tests for lack of convergence
        by comparing the variance between multiple chains to the
        variance within each chain. If convergence has been achieved,
        the between-chain and within-chain variances should be
        identical. To be most effective in detecting evidence for
        nonconvergence, each chain should have been initialized to
        starting values that are dispersed relative to the target
        distribution.
        """
        gelman_rubin_results = pm.diagnostics.gelman_rubin(trace, varnames=mdl.cont_vars)
        print("Gelman Rubin Diagnostic: \n{}\n".format(gelman_rubin_results))

    def effectiveN(self):
        """Measures an estimate of the effective sample size of a set of traces
        effect_n should be greater than 200.
        """
        effect_n_results = pm.diagnostics.effective_n(
            trace, varnames=mdl.cont_vars)
        print("Effective Sample Size: \n{}\n".format(effect_n_results))


class leastSqaures():
    pass

class gaussianProcess():
    pass

class gaussianProcessMarkovCahinMonteCarolo():
    pass