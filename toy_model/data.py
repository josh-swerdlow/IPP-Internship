import numpy as np
import matplotlib.pyplot as plt


class generateFunction:
    """ Initializes a function object

    Attributes:
        * **func** (function obj): a single variable (f(x)) generator based on
            the desired functional output. A func should simply return the
            output required by your function and nothing else. All free
            parameters should be specified and the independent variable must
            be x.
            Example::

                def simpleFunction(x, a, b, c, d):
                    return a * x + b + c + x ** d


        * **indepParam** (list): all the independent parameters in the
            function.

            Note: This should only be x.

        * **freeParam** (dict): a dict of the free parameter names and their
            values
        * **sigma** (int): standard deviation for 'random' normal sampling.
        * **samples** (int): number of samples.
        * **x** (np.array int): array of values in a given range.
        * **noise** (np.array int): array of normal-random scaled values.
        * **y** (np.array int): array of values output based on your function
            with noise included.
    """
    rndst = np.random.RandomState(0)

    def __init__(self, func=None, **freeParams):
        """Creates the function object"""

        if func is None:
            print("Error: func must not be None.")
            raise TypeError

        indepParam = func.__code__.co_varnames[0]
        if indepParam is not 'x':
            print("Error: The independent variable of func must be 'x'.")
            raise ValueError

        if len(func.__code__.co_varnames[1:]) is not len(freeParams):
            print("Error: The number of the free parameters in func and ",
                "the parameters given in freeParams must be the same.")
            raise TypeError

        self.indepParam = indepParam
        self.func = lambda x: func(x, *list(freeParams.values()))
        self.freeParams = freeParams

    def evaluate(self, sigma=0, samples=10):
        if sigma >= 0:
            self.sigma = sigma
        else:
            print("Error: sigma must be greater than or equal to zero")
            raise ValueError

        if samples > 0:
            self.samples = samples
        else:
            print("Error: samples must be greater than zero")
            raise ValueError

        self.x = generateFunction.rndst.choice(
            np.arange(samples), samples, replace=False)
        self.noise = sigma * np.random.normal(0, sigma, samples)
        self.y = self.func(self.x) + self.noise

    def plot(self):
        """Plots the function of y=f(x)"""
        plt.plot(self.x, self.y)
        plt.show()




