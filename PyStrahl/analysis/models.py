# Python file containing our different models:
#   1. Markov Chain Monte Carlo
#   2. Gaussian Process
#   3. Least Square
#   4. Combination of 1 and 2

# Each of these will be given netCDF data groups composed of
# stuff and they will have freeParams etc stuff stuf stuff ugh
# every variable must have its units corrected to SI before hand
import sys

# import pymc3 as pm
import matplotlib.pyplot as plt

from PyStrahl.mirmpfit import mpfit
from PyStrahl.analysis.data import Function, Residual


class Markov_Chain_Monte_Carlo():
    """Initiates a MCMC regression algorithm for the given data"""

    def __init__(self, x_, y_, f_, sigma_info_, priors_,
                 burn=1000, verbose=False):

        self.x = x_
        self.y = y_
        self.model = pm.Model()

        self.fun = f_
        self.sigma_info_ = sigma_info_

        self.priors = dict()
        self.priors_ = priors_

        self.burn = burn

        if verbose:
            self.verbose = verbose

            print("verbose output turned on.")

    # Inititate a basic model #
    def evaluate(self, niter, step):
        DISTRIBUTION_INDEX = 0
        PARAMETERS_INDEX = 1

        with self.model:
            distribution = self.sigma_info_[DISTRIBUTION_INDEX]
            parameters = self.sigma_info_[PARAMETERS_INDEX]

            sigma = distribution('sigma', **parameters)

            for prior, info in self.priors_.items():
                name = prior

                distribution = info[DISTRIBUTION_INDEX]
                parameters = info[PARAMETERS_INDEX]

                print("Adding {} distribution prior ".format(distribution) +
                      "'{}' to the model with parameters ".format(name) +
                      "{}.\n".format(parameters))

                if isinstance(parameters, list):
                    self.priors[name] = distribution(name, *parameters)

                elif isinstance(parameters, dict):
                    self.priors[name] = distribution(name, **parameters)

                else:
                    print("Error: Parameters was not given as a list or dict")

            print(self.priors)
            self.y_exp = self.fun.func(x=self.x, **self.priors)
            print(self.y_exp)

            # self.y_obs = pm.Normal('y_obs', mu=self.y_exp, sd=sigma)

            # Initialize MCMC Algorithm
            # self.trace = pm.sample(niter, step=step, progressbar=True)

    def trace(self):
        """Returns the trace of the MCMC Algorithm"""
        return self.trace

    def mean_parameters(self, burn=None):
        """Returns the average of the parameters except for the burn in"""
        if burn is None:
            burn = self.burn

        mean_parameters = dict()
        for name in self.priors.keys():
            mean_parameters[name] = self.trace[name][burn:]

        return mean_parameters


    # Plotting Routines
    def plot_trace(self):
        """Plots sample histograms and values"""
        pm.traceplot(trace)

    def plot_forest(self):
        """Generates a forest plot of 100*(1-alpha)% credible intervals from a trace"""
        pm.forestplot(trace)

    def plot_posterior(self):
        """Plots posterior density"""
        pm.plot_posterior(trace)

    def plot_fit(self):
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


class Least_Square():

    def __init__(self, x, y, sigma, parameter_info, residual_,
                 main_fn=None, inpt_fn=None, data_fn=None,
                 verbose=False):

        print("\nInitializing Least Square Model...")

        self.res_keys = {'x': x, 'y': y, 'sigma': sigma}

        self.x = x
        self.y = y
        self.sigma = sigma

        if isinstance(parameter_info, list):
            for item in parameter_info:
                if isinstance(item, dict):
                    self.parameter_info = parameter_info
                else:
                    sys.exit("Error: All elements of parameter_info must be of type dict.")
        else:
            sys.exit("Error: parameter_info must be of type list.")

        if not isinstance(residual_, Residual):
            sys.exit("Error: residual_ is not a Residuals object.")

        if main_fn is not None:
            residual_.set(main_fn=main_fn)

        if inpt_fn is not None:
            residual_.set(inpt_fn=inpt_fn)

        if data_fn is not None:
            residual_.set(data_fn=data_fn)

        self.residual_ = residual_

        self.verbose = verbose

    def __str__(self):

        return str(self.__dict__)

    def fit(self, coeffs, parinfo=None, stats=False):

        if parinfo is None:
            if self.verbose:
                print("Using original initialized parameter info.")

            parinfo = self.parameter_info

        else:
            if not isinstance(parinfo, list):
                sys.exit("Error: parainfo must be of type list.")

            else:
                for item in parinfo:
                    if not isinstance(item, dict):
                        sys.exit("Error: All elements of parinfo must be of type dict.")

        print("Running mpfit...")
        if self.verbose:
            print("\tInitial guess: {}".format(coeffs))
            print("\tResidual keys: {}".format(self.res_keys.keys()))
            print("\tParameter info: {}".format(parinfo))

        mpfit_ = mpfit(self.residual_.mpfit_residual,
                       residual_keywords=self.res_keys, parinfo=parinfo,
                       quiet=False)

        print("mpfit has completed executing...")
        print("\tError Message: {}".format(mpfit_.errmsg))
        print("\tStatus: {}".format(mpfit_.statusString()))

        index = 0
        for param, error in zip(mpfit_.params, mpfit_.perror):
            print("params[{}]: {} with perror {}".format(index, param, error))
            index += 1

        # Reinitialize the spline object with the fitted knot coeffs
        self.residual_.re_init_spline(y=mpfit_.params)

        # Calculate some statistics for storage
        weighted_residual = self.residual_.residual()
        weighted_residual_sq = self.residual_.residual_squared()

        unweighted_residual = self.residual_.residual(weighted=False)
        unweighted_residual_sq = self.residual_.residual_squared(weighted=False)

        chi_sq = self.residual_.chi_squared()
        reduce_chi_sq = self.residual_.reduced_chi_squared(v=mpfit_.dof)

        self.statistics = {'Weighted Residual': weighted_residual,
                           'Weighted Residual Squared': weighted_residual_sq,

                           'Unweighted Residual': unweighted_residual,
                           'Unweighted Residual Squared': unweighted_residual_sq,

                           'Chi Squared': chi_sq,
                           'Reduced Chi Squared': reduce_chi_sq}

        if stats:
            print("Unweighted Residual: {}".format(unweighted_residual))
            print("Unweighted Residual Squared: {}".format(unweighted_residual_sq))

            print("Weighted Residual: {}".format(weighted_residual))
            print("Weighted Residual Squared: {}".format(weighted_residual_sq))

            print("Chi Squared: {}".format(chi_sq))
            print("Reduced Chi Squared: {}".format(reduce_chi_sq))

        self.mpfit_ = mpfit_

        return mpfit_

    def plot_sigmas(self, x=None, sigma=None, hold=False, save=None):
        if x is None:
            x = self.x

        if sigma is None:
            sigma = self.sigma

        else:
            print("No sigma value given or found within {}".format(self))

        plt.title("Signal Error")
        plt.scatter(x, sigma)
        plt.legend(["Weightings"])

        if not hold:
            plt.show()

    def plot_error(self, hold=False, save=None):
        plt.figure(self.fun_.name)
        plt.title(self.fun_.name)

        if hasattr(self, 'mpfit_') and hasattr(self, 'residual_'):
            x_knots = self.residual_.spline_.x_knots
            y_knots = self.mpfit_.params
            perror = self.mpfit_.perror

            plt.errorbar(x_knots, y_knots,
                        fmt='o-', ecolor='r', yerr=perror)
        else:
            print("No mpfit_ or residual_ object found within {}".format(self))

        plt.legend(["Fit with error"])

        if not hold:
            plt.show()

    def plot_fit(self, hold=False, save=None):

        # Title formatting
        plt.figure("Fitted Plot")
        plt.title("Fitted f(x) = {}".format(self.fun_.equation))

        # Initiating the spline interpolation on fitting points
        # spl = self.spline_model.spline_function(self.fit.params)

        plt.plot(self.x, self.y, 'k.')

        if hasattr(self, 'mpfit_') and hasattr(self, 'residual_'):
            x_knots = self.residual_.spline_.x_knots
            y_knots = self.mpfit_.params
            perror = self.mpfit_.perror
            fit = self.residual_.spline_(self.x)

            plt.plot(self.x, fit, 'r')

            plt.errorbar(x_knots, y_knots,
                         fmt='x', elinewidth=1.5, yerr=perror)
        else:
            print("No mpfit_ or residual_ object found within {}".format(self))

        plt.legend(["Experimental Signal", "Fitted Estimate", "Knots with error"])
        plt.show()

    def plot_residual(self, weighted=True, hold=False, save=None):

        figure_str = "Residual"
        if weighted:
            figure_str = "Weighted " + figure_str
        else:
            figure_str = "Unweighted " + figure_str

        if hasattr(self, 'statistics'):
            residuals = self.statistics[figure_str]

        plt.title(figure_str)
        plt.scatter(self.x, residuals)
        plt.legend({figure_str})

        if not hold:
            plt.show()

class Gaussian_Process():
    pass


class gaussianProcessMarkovCahinMonteCarolo():
    pass
