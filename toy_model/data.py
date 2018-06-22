import numpy as np
import matplotlib.pyplot as plt


class generateFunction:
    """ Create a function object that stores certain features of the
    given function; such as, the given free parameters. In addition,
    one can evaluate the function over a range and add noise sampled
    from given normal distribution.

    """
    rndst = np.random.RandomState(0)

    def __init__(self, func=None, **freeParams):
        """Initializes the function object

        Attributes:
            * **indepParam** (str): name of indepedent variable
            * **func** (function obj): function that has the free
                parameters input.

                Example:

                .. code-block:: python

                    def simpleFunction(x, m, b):
                        return m * x + b

                    func = lambda x: simpleFunction(x, 1, 2)
            * **freeParams** (dict): a dict of the free parameter names and
                their values

        Raises:
            * **TypeError**: If given None for function
            * **TypeError**: If not given enough freeParam for the given func
            * **ValueError**: If the independent variable of func is not 'x'

        """

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
        """Evaluates the function across a sample of data

        Arguments:
            * **sigma** (int): the standard deviation
            * **samples** (int): number of samples for data

        Attributes:
            * **sigma** (int): the standard deviation
            * **samples** (int): number of samples for data
            * **x** (np.array int): array of values in a given range.
            * **noise** (np.array int): array of normal-random scaled values.
            * **y** (np.array int): array of values output based on your
                function with noise included.

        Raises:
            * **ValueError**: If sigma is less than zero
            * **ValueError**: If samples is less than zero

        """
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




