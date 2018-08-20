# We are going to make a toy model MCMC that fits a 'linear' data
# set with noise with two parameters alpha and beta which are
# normally distributed

import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import scipy as sp

from data import Function, Splines, Residuals
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from mirmpfit import mpfit

from models import Least_Squares, Markov_Chain_Monte_Carlo

rndst = np.random.RandomState(0)

def init_data(func, samples=10, sigma=0, plot=False):
    """ Intializes noisy data given a single variable function.

    Creates a normal noisy data signal with specified number of samples.

    Args:
        * **func** (lambdaType): a generator based on the desired functional output.
        * **samples** (int): number of samples.
        * **sigma** (int): standard deviation for 'random' normal sampling.
        * plot (bool): *GET RID OF*

    Returns:
        A pandas dataFrame of the x values and y values evaluated given the
        argument func.

        i.e.) y = func(x)

    Raises:
        None.
    """

    # Init DataFrame
    df = pd.DataFrame(
        {'x': rndst.choice(np.arange(samples), samples, replace=False)})

    # Generate noise
    noise = sigma * np.random.normal(0, sigma, samples)

    # Calculate y with noise added
    df['y'] = func(df['x']) + noise

    if plot:
        sns.lmplot(x='x', y='y', data=df, fit_reg=False, legend=True)
        plt.show()

    return df

def mcmc_model():

    x_ = np.linspace(0, 5, 10)

    m, b = 1, 0
    f_ = Function.linear(m, b)
    f = f_.evaluate(x=x_)

    background_scale, intensity_scale = 1, 0
    noise_ = Function.noise(background_scale, intensity_scale,
                            parameterization=f_)
    noise = noise_.evaluate(x=x_)

    y_ = f + noise

    # Priors for unknown model data
    # These represent the distributions that will
    # be looked into when tuning the parameters
    p1_name = 'm'
    p1_distribution = pm.Normal
    p1_parameters = {'mu': 0, 'sd': 5}
    p1_info = [p1_distribution, p1_parameters]

    p2_name = 'b'
    p2_distribution = pm.Normal
    p2_parameters = {'mu': 0, 'sd': 5}
    p2_info = [p2_distribution, p2_parameters]

    priors_ = {p1_name: p1_info, p2_name: p2_info}

    sigma_distribution = pm.HalfNormal
    sigma_parameters = {'sd': 5}
    sigma_info_ = [sigma_distribution, sigma_parameters]

    mdl = markov_chain_monte_carlo(x_, y_, f_, sigma_info_, priors_,
                                   verbose=True)
    niter = 1000
    step = pm.step_methods.Metropolis
    mdl.evaluate(niter, step)

    # Inititate a basic model #
    # mdl = pm.Model()
    # with mdl:

        # print('Added alpha to model...')
        # sigma = pm.HalfNormal('sigma', sd=5)
        # print('Added sigma to model...')
        # if beta:
        #     beta = pm.Normal('beta', mu=0, sd=10)
        #     print('Added beta to model...')
        # if gamma:
        #     gamma = pm.Normal('gamma', mu=0, sd=10)
        #     print('Added gamma to model...')

        # # This is the expected value of our function
        # # so it does not have any noise.
        # y_exp = alpha * (df['x']) + beta * (df['x'])**2 + gamma

        # # This is the observed value of our function
        # # based on what we expect the distribution of values to be
        # y_obs = pm.Normal('y_obs', mu=y_exp, sd=sigma, observed=df['y'])

        # # Select MCMC Algorithm #
        # step = pm.step_methods.Metropolis()
        # # step = pm.step_methods.HamiltonianMC()
        # # step = pm.step_methods.NUTS()

        # # Initialize MCMC Algorithm
        # trace = pm.sample(20000, step=step)

def lstsq_model():
    # Establish some space to sample over
    start, stop = 0, 5

    data_pts = 50
    x = np.linspace(start, stop, data_pts)

    # Generate a function and sample over that function
    f_ = Function.quadratic(1, 3, 0)
    f = f_.evaluate(x=x)

    # Generate noise over the same range as the function
    background_scale, intensity_scale = 1, 0
    noise_ = Function.noise(background_scale, intensity_scale,
                            parameterization=f_)
    noise = noise_.evaluate(x=x)

    # Add noise to function to create an experimental signal
    # Change so that there is y_ (function object), y (data pts),
    # and y_knots (y_ evaluated over x_knots)
    # HELPFUL because we can make a sigma_knots easily
    y = f + noise

    # Evaluate error of the signal
    sigma = np.sqrt(background_scale**2 + np.abs(intensity_scale * y))

    knot_pts = 5
    x_knots = np.linspace(start, stop, knot_pts)
    y_knots = f_.evaluate(x=x_knots) + noise_.evaluate(x=x_knots)
    spline_ = Splines.linear_univariate_spline(x_knots, y_knots)

    res_keys = {'x': x, 'y': y, 'sigma': sigma}
    residual_ = Residuals(spline_, **res_keys)

    mdl = least_squares(x, y, sigma, residual_, f_)

    coeffs = [list()]

    mdl.fit(coeffs)


    # mdl.plot_sigmas()
    # mdl.plot_error()
    # mdl.plot_fit()
    # mdl.plot_residual(weighted=False)


if __name__ == '__main__':
    # mcmc_model()
    lstsq_model()








