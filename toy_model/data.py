import numpy as np
import matplotlib.pyplot as plt


class generateFunction:
    """ Initializes a function object

    Attributes:
        * **func** (lambdaType): a generator based on the desired functional output.
        * **indepParam** (list): all the independent parameters in the function.
        * **depenParam** (list): all the dependent paramters in the function.
        * **sigma** (int): standard deviation for 'random' normal sampling.
        * **samples** (int): number of samples.
        * **x** (np.array int): list of values in a given range.
        * **noise** (np.array int): list of normal-random scaled values.
        * **y** (np.array int):
    """
    rndst = np.random.RandomState(0)

    def __init__(self, func=None, sigma=0, samples=10):
        if func is not None:
            self.func = func
            self.indepParam = func.__code__.co_varnames[0]
            self.depenParam = func.__code__.co_varnames[0:]
            self.samples = samples
            self.sigma = sigma

            self.x = generateFunction.rndst.choice(
                np.arange(samples), samples, replace=False)
            self.noise = sigma * np.random.normal(0, sigma, samples)
            self.y = self.func(self.x) + self.noise
        else:
            raise TypeError

    def plot(self):
        """Plots the function"""
        plt(self.x, self.y)
        plt.show()




