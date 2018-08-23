import matplotlib.pyplot as plt
import numpy as np

# Turn this into a results object that upon initialization
# loads the entire dictionary into a parameter and then has
# different methods for plotting anything necessary

# Add class method which loops through and creates numerous
# result objects so one can iterate through them and plot
# everything they need (will require good error handling!)

# Show be able to choose the backend

# Develop something that parses kwargs for the figure and the plot
# figure: give the function a dict of expected plt.method
# i.e. legend, title, xlim and then they have the values youd like
# to set. Then we do a try, except block on the key (see weighted_residual)
# This try, except should be turned into a function which takes the dict
# the method, and the value
# As well, there should be a set of initial configurations
# that one can pass to the results object
# these will stay constant and be passed to ever plot method
# There will be an attribute for both the plot and figure settigns
# so one can set defaults for all of the plots at the same time

# This should really be abstracted into two classes
# One should be the base class which has the try,except function
# the initialization, the dict parsing,
# The other should just handle results for this project and make
# specific plots that I need for this


def sigmas(x, sigma, show=False, fn=None):

    plt.title("Signal Error")

    plt.scatter(x, sigma)

    plt.legend(["Weightings"])

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def error(x_knots, y_knots, perror, show=False, fn=None):

    plt.figure("Error")
    plt.title("Error")

    plt.errorbar(x_knots, y_knots,
                fmt='o-', ecolor='r', yerr=perror)

    plt.legend(["Fit with error"])

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def signal_fit(time, signal, fit_val,
                    x_knots, y_knots, perror,
                    show=False, fn=None):

    # Title formatting
    plt.figure("Fitted Plot")
    plt.title("Fitted f(x)")

    plt.plot(time, signal, time, fit_val, 'k.')

    plt.errorbar(x_knots, y_knots,
                fmt='o-', ecolor='r', yerr=perror)

    plt.legend(["Experimental Signal", "Fitted Estimate", "Knots with error"])

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def weighted_residual_sq(x, weighted_residual_squared, show=False, fn=None):
    plt.plot(x, weighted_residual_squared)
    plt.legend(['weighted_residual_squared'])
    plt.xlabel("Rho")

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def residual_sq(x, residual_squared, show=False, fn=None):
    plt.plot(x, residual_squared)
    plt.legend(['residual_squared'])
    plt.xlabel("Rho")

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def weighted_residual(x, weighted_residual, args, plot_kwargs=None,
                      show=False, fn=None, fig_kwargs=None):

    if plot_kwargs is None:
        plot_kwargs = dict()

    if fig_kwargs is None:
        fig_kwargs = dict()

    try:
        figure = fig_kwargs['figure']
    except KeyError:
        figure = 0
    plt.figure(figure)

    plt.plot(x, weighted_residual, *args, **plot_kwargs)

    try:
        legend = fig_kwargs['legend']
    except KeyError:
        legend = []

    plt.legend(legend)

    try:
        title = fig_kwargs['title']
    except KeyError:
        title = ""

    plt.title(title)

    plt.xlabel("Time (s)")

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


def residual(x, residual, show=False, fn=None):
    plt.plot(x, residual)
    plt.legend(['residual'])
    plt.xlabel("Rho")

    if fn is not None and isinstance(fn, str):
        plt.savefig("./plots/" + fn)

    if show:
        plt.show()


if __name__ == '__main__':
    results("test_summary.json")