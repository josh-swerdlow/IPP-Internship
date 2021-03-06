########################################
# File name: math.py                   #
# Author: Joshua Swerdow               #
# Date created: 8/21/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

__doc__ = """
          This module contains custom made objects which are specifically
          created to be passed as arguments into high level methods. These
          objects (Function, Residual, and Splines) mostly wrap around the
          obvious method, class, or mathematical object and add metadata,
          methods, and class method initializers for specific use cases.
          """

__author__ = "Joshua Swerdow"


import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from PyStrahl.utils import math
from PyStrahl.core import strahl
from scipy.interpolate import InterpolatedUnivariateSpline

FLOAT_MIN_VAL = sys.float_info.min
FLOAT_EPSILON = sys.float_info.epsilon
FLOAT_ABS_MIN = FLOAT_MIN_VAL * FLOAT_EPSILON


# Before I finish commenting, let's:
#   Work out the random noise problem
#   Bring the magic methods up to my standards
#
class Function:
    """Create a Function object that stores features of the
    given single variable function.

    """
    rndst = np.random.RandomState(0)

    def __init__(self, name, math, func,
                 parameterization=None, eqtn=None, data=None,
                 **free_params):
        """Initializes the Function object

        Attributes:
            name (str): a custom made string name for the object
            math (str): A variable format string of the equation that
                represents the function.

                Example: y = m * x + b --> math = "mx+b" or "m*x+b"
            func ()

            * **indep_param** (str): name of indepedent variable
            * **func** (Function obj): function that has the free
                parameters input.

                Example:

                .. code-block:: python

                    def simpleFunction(x, m, b):
                        return m * x + b

                    func = lambda x: simpleFunction(x, 1, 2)
            * **free_params** (dict): a dict of the free parameter names and
                their values

        Raises:
            * **TypeError**: If not given enough freeParam for the given func

        """
        if name is None:
            name = "No name provided."

        self.name = name
        self.general_fun = func

        if math is None:
            math = "('{}')".format(name)

        if not (math.startswith("(") or math.startswith("[")):
            math = "(" + math

        if not (math.endswith(")") or math.endswith("]")):
            math = math + ")"

        if eqtn is None:
            equation = math
        else:
            equation = eqtn

        if not callable(func):
            sys.exit("Error: func is NoneType.")

        parameter_names = func.__code__.co_varnames

        number_of_parameters = len(parameter_names)

        free_param_vals = list(free_params.values())

        if number_of_parameters == 0:
            print("Error: no parameters given.")

        if number_of_parameters == 1:
            print("Only one parameter given...")

            if parameter_names[0] == "x":
                print("Taking the independent variable as {}"
                      .format(parameter_names[0]))

                indep_param = parameter_names[0]
                if len(free_params) == 0:
                    free_params = dict()

            else:
                print("Assuming constant function. First parameter is not 'x'")
                indep_param = "_"
                free_params = free_params

        elif number_of_parameters > 1:
            indep_param = parameter_names[0]

            print("Taking the independent variable as {}"
                  .format(indep_param))

            if (number_of_parameters - 1) == len(free_params):
                free_params = {name: free_params[name] for name in parameter_names[1:]}

                print("Taking the free parameters as {}".format(free_params))

            else:
                print("Error: The number of the free parameters in func and ",
                    "the parameters given in freeParams must be the same.")

                raise TypeError

        if parameterization is None:
            if (number_of_parameters - 1) == len(free_params):
                fun = lambda x: func(x, *free_param_vals)
            else:
                fun = lambda x: func(x)

        elif isinstance(parameterization, Function):
            print("Parameterizing f({}) = {};"
                  .format(indep_param, math))

            fun = (lambda x:
                   func(parameterization.func(x), *free_param_vals))

            parameterization_math_str = parameterization.math[1:-1]
            parameterization_eqtn_str = parameterization.equation[1:-1]

            math = (math.replace(indep_param, "[{function}]")
                        .format(function=parameterization_math_str))

            equation = (equation.replace(indep_param, "[{function}]")
                        .format(function=parameterization_eqtn_str))

            print("\twith {}({}) = {}"
                 .format(indep_param, parameterization.indep_param,
                         parameterization_math_str))

            indep_param = parameterization.indep_param
            free_params.update(parameterization.free_params)

        else:
            print("parameterization is not a function object")

        self.func = fun
        self.math = math

        if data is None or not isinstance(data, (list, np.ndarray)):
            data = 0
        self.data = data

        cp_free_params = copy(free_params)
        sorted_free_params = sorted(cp_free_params.keys(), key=len,
                                    reverse=True)

        for param in sorted_free_params:
            index = str(sorted_free_params.index(param))

            equation = equation.replace(param, "{{'{val}'}}".format(val=index))
            cp_free_params["'" + index + "'"] = cp_free_params.pop(param)

        equation = equation.format(**cp_free_params)

        self.equation = equation

        self.indep_param = indep_param

        self.free_params = free_params

        self.evaluated = False

        print("Generating function {}:".format(fun))
        print("f({}, {}) = {}\n".format(indep_param, free_params, math))

    def __add__(self, other):
        """
        Overwrites add function to work with function objects, np arrays,
        and lists.
        """

        if isinstance(self, Function) and isinstance(other, Function):
            return self._func_add(other)

        if isinstance(other, (np.ndarray, list)):
            name = "{}-{}".format(self.name, "with evaluated data")
            math = "{}+{}".format(self.math, "(data)")

            if isinstance(other, list):
                other = np.asarray(other)

            if len(other.shape) == 1:
                data = self.data + other
            else:
                print("Cannot add object of shape {}".format(other.shape))

            fun = Function(name, math, self.func,
                           data=data,
                           **self.free_params)

            return fun
        else:
            print("Adding like normal")
            return self + other

    def _func_add(self, func):
        """
        Creates a new function object which is the sum of the arguments.
        The arguments must be function objects that depend on the same
        variable.

        Parameters:
            * **func** [function object]: A function object that will be
                summed with another function

        Returns:
            A new function object
        """
        print("Creating a summed function...")

        function = lambda x: self.func(x) + func.func(x)

        func_params = func.free_params
        func_param_keys = func.free_params.keys()
        self_param_keys = self.free_params.keys()

        # Changes free_parameters with the same name
        for self_key in self_param_keys:
            if self_key in func_param_keys:
                new_key = self_key + "'"

                func.math = func.math.replace(self_key, new_key)

                func_params[new_key] = func_params.pop(self_key, None)

        self.free_params.update(func_params)

        name = "{}-{}".format(self.name, func.name)

        math = "{}+{}".format(self.math, func.math)

        Fun = Function(name, math, function, **self.free_params)

        return Fun

    def __str__(self):
        """
        Wraps around the native python str() function

        Returns:
            A mathematical formula representing the function
        """

        return self.equation

    def __iadd__(self, func):
        """
        Overwrites the iadd method (a += b) to operate with function
        objects
        """
        if not isinstance(self, Function) and not isinstance(func, Function):
            return self.__iadd__(func)

        return self.__add__(func)

    def evaluate(self, start=None, stop=None, steps=None, x=None,
                 plot=False, scatter=False, verbose=False):
        """Evaluates the function across a sample of data

        Arguments:
            * **samples** (int): number of samples for data

        Attributes:
            * **x** (np.ndarray of int): array of values in a given range.
            * **y** (np.ndarray of int): array of values output based on your
                function with noise included.

        Raises:
            * **ValueError**: If start is greater than stop
            * **ValueError**: If step is less than or equal to zero

        """
        if (start is None or stop is None) and x is None:
            if start is None:
                start = 0

            if stop is None:
                stop = 50

            print("Warning: Requires either start and stop or x")
            print("Defaulting to start = 0 and stop = 50")

        if start is not None and stop is not None:
            if start > stop:
                print("Error: start must be greater than stop")
                raise ValueError

            if steps is None:
                steps = 50

            if steps <= 0:
                print("Error: step must be greater than 0")
                raise ValueError

            x = np.linspace(start, stop, steps)

        if x is not None:
            if not isinstance(x, (list, np.ndarray)):
                print("Error: x is not of type list or np.ndarray")
                raise TypeError

        self.x = x
        function_data = np.asarray(self.func(self.x))

        self.y = np.add(function_data, self.data)

        if verbose:
            spaces = " " * (len(self.indep_param) + 3)
            print("\nf({}) = {}\n{} = {}\n"
                  .format(self.indep_param, self.math,
                          spaces, self.equation))
            print("{} = {}\n\n".format(spaces, self.y))

        if plot:
            self.plot()

        if scatter:
            self.scatter()

        self.evaluated = True

        return self.y

    def plot(self, x=None):
        """Plots the function of y=f(x)"""
        print("Plotting f({}) = {}".format(self.indep_param, self.math))

        title = self.name

        if x is not None:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)

            y = self.func(x)

            plt.plot(x, y)

        else:
            if hasattr(self, 'x') and hasattr(self, 'y'):
                plt.plot(self.x, self.y)

            else:
                print("This object has not been evaluated (see function.evaluate)")

        plt.title(title)
        plt.show()

    def scatter(self, x=None):
        """Scatter plots the function of y=f(x)"""
        print("Plotting f({}) = {}".format(self.indep_param, self.math))

        title = self.name

        if x is not None:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)

            y = self.func(x)

            plt.scatter(x, y)

        else:
            if hasattr(self, 'x') and hasattr(self, 'y'):
                plt.scatter(self.x, self.y)

            else:
                print("This object has not been evaluated (see function.evaluate)")

        plt.title(title)
        plt.show()

    def coefficients(self):

        return self.free_parameters

    @classmethod
    def linear(cls, m, b, parameterization=None):
        """Creates a linear Function object of the form f(x) = m * x + b

        Args:
            m (int or float): Slope of the function
            b (int or float): y intercept of the function
            parameterization (:obj:Function): A parameterization function
                to initiate a parameterized linear function.

                Example:

                    If our linear function is f(x) = 5x + 1
                    and our parameterization function is g(x) = 2x - 1;

                    then we will parameterize f(x) with g(x) as f(g(x))
                    such that f(g(x)) = f(x) = 5 (2x - 1) + 1 = 10x - 4

        Returns:
            A Function object for a linear mathematical function (with a
            parameterization if specified).

        """
        print("Creating a linear function...")

        name = "linear"
        if parameterization is not None:
            name = name + " with parameterization"

        math = "mx + b"

        return cls(name, math, Function._linFunc,
                   parameterization=parameterization,
                   m=m, b=b)

    @classmethod
    def quadratic(cls, a, b, c, parameterization=None):
        """Creates a quadratic Function object of the form
        f(x) = a*x^2 + b*x + c

        Args:
            a (int or float): second order coefficient
            b (int or float): first order coefficient
            c (int or float): zeroth order coefficient
            parameterization (:obj:Function): A parameterization function
                to initiate a parameterized quadratic function.

                Example:

                    If our quadratic function is f(x) = 5x^2 + 1
                    and our parameterization function is g(x) = 2x;

                    then we will parameterize f(x) with g(x) as f(g(x))
                    such that f(g(x)) = f(x) = 5 (2x)^2 + 1 = 20x^2 + 1

        Returns:
            A Function object for a quadratic mathematical function (with a
            parameterization if specified).

        """
        print("Creating a quadratic function...")

        name = "quadratic"
        if a != 0:
            name = "2nd order polynomial"
            if parameterization is not None:
                name = name + " with parameterization"

        elif b != 0:
            name = "1st order polynomial"
            if parameterization is not None:
                name = name + " with parameterization"

        else:
            name = "0th order polynomial"
            if parameterization is not None:
                name = name + " with parameterization"

        math = "ax^2 + bx + c"

        return cls(name, math, Function._quadFunc,
                   parameterization=parameterization,
                   a=a, b=b, c=c)

    @classmethod
    def constant(cls, a):
        """Creates a constant Function object of the form f(x) = a

        Note that constant functions cannot be parameterized.

        Args:
            a (int or float): The constant value of the function

        Returns:
            A Function object for a constant mathematical function.
        """
        print("Creating a constant function...")

        if a is None:
            a = 0

        name = "constant"
        math = "a"

        return cls(name, math, Function._constFunc, a=a)

    @classmethod
    def noise(cls, sigma_background, scale_factor, parameterization=None):
        """Creates a noise Function object of the form
        f(Intensity) = N(0, sigma_background) + N(0, scale_factor * Intensity)

        Note: N(loc, scale) is the syntax used to describe a normal distribution
        centered at loc with a width of scale.

        Args:
            sigma_background (int or float): Background noise signal scaling
            scale_factor (int or float): Intensity noise signal scaling
            parameterization (:obj:Function): A parameterization function
                to initiate a parameterized noise function.

                Example:

                    If our noise function is
                    f(Intensity) = N(0, 1) + N(0, 2 * Intensity)
                    and our parameterization function is
                    g(Intensity) = (2 * Intensity) - 1;

                    then we will parameterize f(Intensity) with g(Intensity)
                    as f(g(Intensity)) such that
                    f(g(Intensity)) =
                       f(Intensity) = N(0, 1) + N(0, 2 * (2 * Intensity - 1))
                                    = N(0, 1) + N(0, 4 * Intensity - 2)

        Returns:
            A Function object for a mathematical function to model noise (with a
            parameterization if specified).

        """
        print("Creating a noise function...")

        name = "noise"
        if sigma_background != 0 and scale_factor == 0:
            name = "constant background noise"

        elif scale_factor != 0:
            name = name + " scaled with intensity"

            if parameterization is not None:
                name = "parameterized " + name

        math = "N(0, sig) + N(0, A * intensity)"

        return cls(name, math, Function._noiseFunc,
                   parameterization=parameterization,
                   sig=sigma_background,
                   A=scale_factor)

    @staticmethod
    def _linFunc(x, m, b):
        """A linear function

        Args:
            x (int or float array): The independent variable
            m (int or float): The slope of the function
            b (int or float): The y intercept of the function

        Returns:
            The value of some linear function for a given vector of values
        """
        return m * x + b

    @staticmethod
    def _quadFunc(x, a, b, c):
        """A quadratic function

        Args:
            x (int or float array): The independent variable
            a (int or float): The second order coefficient
            b (int or float): The first order coefficient
            c (int or float): The zeroth order coefficient

        Returns:
            The value of some quadratic function for a given vector of values.
        """
        return a * x**2 + b * x + c

    @staticmethod
    def _constFunc(a):
        """A constant function

        Args:
            a (int or float): The constant value of the function

        Returns:
            As this is a constant function, it always returns itself.
        """
        return a

    @staticmethod
    def _noiseFunc(intensity, sig, A):
        """A noise function that scales with some intensity parameter

        Args:
            intensity (int or float array): The signal intensity
            sig (int or float): The background noise signal scaling
            A (int or float): The intensity noise signal scaling

        Returns:
            A simple model for the noise of a given vector of intensities
        """
        return (np.random.normal(0, sig, len(intensity)) +
                np.random.normal(0, np.sqrt(A * intensity), len(intensity)))


# Before I start commenting, let's:
#   Figure out if there is a good and general way to create Residual object regardless or D and v splines!

class Residual:

    def __init__(self, D_spline_, v_spline_, main_fn, inpt_fn,
                 data_fn, charge_index, integration_dim,
                 fit=None, x=None, y=None, sigma=None,
                 strahl_verbose=False, verbose=False):
        """
        Initializes a STRAHL_Residual object which can calculate
        various parameters (weighted_residual, unweight_residual,
        residual, chi_squared, and reduced chi_squared) and be integrated
        into mpfit with the evaluate method.
        """
        print("\nInitializing Residual object...")

        if not isinstance(D_spline_, Splines):
            sys.exit("D_spline_ must be Splines object.")

        self.D_spline_ = D_spline_

        if not isinstance(v_spline_, Splines):
            sys.exit("v_spline_ must be Splines object.")

        self.v_spline_ = v_spline_

        # Test x_knots are same size and same values
        if len(self.v_spline_.x_knots) != len(self.D_spline_.x_knots):
            sys.exit("x_knots for D and v must be same length.")

            if self.v_spline_.x_knots != self.D_spline_.x_knots:
                sys.exit("x_knots for D and v must be the same.")

        # Test x_knots are an even size
        if len(self.D_spline_.x_knots) % 2 != 0:
            sys.exit("x_knots must have an even number of elements.")

        self.x_knots = self.D_spline_.x_knots
        self.numb_knots = len(self.x_knots)

        # Test y_knots are same size as each other and x_knots
        if len(self.v_spline_.y_knots) != len(self.D_spline_.y_knots):
            sys.exit("Y_knots for D and v must be same length.")

            if len(self.v_spline_.y_knots) != len(self.v_spline_.x_knots):
                sys.exit("v_spline's y_knots and x_knots must be same size.")

            if len(self.D_spline_.y_knots) != len(self.D_spline_.x_knots):
                sys.exit("D_spline's y_knots and x_knots must be same size.")

        self.Dy_knots = self.D_spline_.y_knots
        self.vy_knots = self.v_spline_.y_knots

        main_path = os.path.join("./param_files", main_fn)
        if not os.path.isfile(main_path):
            sys.exit("{} is not a valid main file.".format(main_fn))

        self.main_fn = main_fn

        # We do not check if inpt_fn is a valid file, since it can be created!!
        # Please be careful!
        if verbose:
            print("Not checking if {} is created or not.".format(inpt_fn))

        self.inpt_fn = inpt_fn

        result_path = os.path.join("./result", data_fn)
        if not os.path.isfile(result_path):
            sys.exit("{} is not a valid result file.".format(data_fn))

        self.data_fn = data_fn

        if charge_index < 0:
            sys.exit("charge_index must be >= 0.")

        self.charge_index = charge_index

        if not isinstance(integration_dim, str):
            sys.exit("integration_dim must be a string instance.")

        self.integration_dim = integration_dim

        if fit is None:
            print("WARNING: fit is NoneType. Use set() to assign a fit.")
            print("Using strahl fitting algorithm by default.")
            self.fit = self.strahl_fit
        else:
            self.fit = fit

        if x is None:
            print("x is NoneType. Use set() to assign a value.")
        else:
            self.x = x

        if y is None:
            print("y is NoneType. Use set() to assign a value.")
        else:
            self.y = y

        if sigma is None:
            print("sigma is NoneType. Use set() to assign a value.")
        else:
            self.sigma = sigma

        self.verbose = verbose
        if self.verbose:
            print("Executing with verbose output.")

        self.strahl_verbose = strahl_verbose
        if self.strahl_verbose:
            print("Executing strahl with verbose output!")

    def __str__(self):
        res_dict = self.__dict__
        res_str = "Residual attributes:\n"

        for key, item in res_dict.items():
            if isinstance(item, Splines):
                res_str += "\t{}: Splines Object\n".format(key)
                res_str += "\t\t{}\n".format(item)
            else:
                res_str += "\t{}: {}\n".format(key, item)

        return res_str

    def set(self, D_spline_=None, v_spline_=None,
            main_fn=None, inpt_fn=None, data_fn=None,
            fit=None, x=None, y=None, sigma=None,
            fit_val=None, coeffs=None):

        # Handle setting D_spline_
        if D_spline_ is not None:
            if isinstance(D_spline_, Splines):
                print("Assigned spline_ to {}".format(D_spline_.spline_class))
                self.D_spline_ = D_spline_

            else:
                print("D_spline_ must be a Spline object.")

        # Handle setting v_spline_
        if v_spline_ is not None:
            if isinstance(v_spline_, Splines):
                print("Assigned spline_ to {}".format(v_spline_.spline_class))
                self.v_spline_ = v_spline_

            else:
                print("v_spline_ must be a Spline object.")

        # Handle setting main_fn
        if main_fn is not None:
            # Main parameter files should be in ./param_files
            main_path = os.path.join("./param_files", main_fn)

            if not os.path.isfile(main_path):
                print("{} is not a valid main file.".format(main_fn))

            else:
                self.main_fn = main_fn

        # Handle setting inpt_fn
        if inpt_fn is not None:
            # Input files should be in main directory .
            if self.verbose:
                print("Not checking if {} is created or not.".format(inpt_fn))

            self.inpt_fn = inpt_fn

        # Handle setting data_fn
        if data_fn is not None:
            # Result files should be in ./result
            result_path = os.path.join("./result", data_fn)

            if not os.path.isfile(result_path):
                print("{} is not a valid result file.".format(data_fn))

            else:
                self.data_fn = data_fn

        # Handle setting fit
        if fit is not None:
            if callable(fit):
                print("Assigned fit to {}".format(fit))
                self.fit = fit

            else:
                print("fit must be a function.")

        # Handle setting x
        if x is not None:
            print("Assigned x to {}".format(x))
            self.x = x

        # Handle setting y
        if y is not None:
            print("Assigned y to {}".format(y))
            self.y = y

        # Handle setting sigma
        if sigma is not None:
            print("Assigned sigma to {}".format(sigma))
            self.sigma = sigma

        # Handle setting fit_val
        if fit_val is not None:
            print("Assigned fit_val to {}".format(fit_val))
            self.fit_val = fit_val

        # Handle setting coeffs
        if coeffs is not None:
            print("Assigned coeffs to {}".format(coeffs))
            self.coeffs = coeffs

    def mpfit_residual(self, coeffs, x, y, sigma):
        fit = self.fit(coeffs=coeffs, x=x)

        # Protect from divide by 0 errors
        sigma[sigma == 0.0] = FLOAT_ABS_MIN

        residual = np.divide(y - fit, sigma)

        return {'residual': residual}

    def residual(self, y=None, sigma=None, fit_val=None, coeffs=None,
                 x=None, weighted=True):
        """Calculates either the weighted or unweighted residual.

        If any of the keywords are a boolean instances of True, then
        this method expects there to be an attribute assigned to the
        object for that keyword which it can use. Otherwise, if any of
        the keywords are not a boolean instances of True, this method
        expects the input value to be given in the correct form.

        If fit_val is True or specified, then coeffs and x should not
        be specified or True. Otherwise, coeffs and x should be specified
        or True.

        If weighted is False, sigma should not be specified or True.

        Args:

        Returns:


        """
        if y is True:
            if hasattr(self, 'y'):
                y = self.y
        elif y is None:
            sys.exit("{} has no attribute 'y'".format(self))

        if weighted is not True:
            sigma = 1
        elif sigma is True:
            if hasattr(self, 'sigma'):
                sigma = self.sigma
        elif sigma is None:
            sys.exit("{} has no attribute 'sigma'".format(self))

        if fit_val is True:
            if hasattr(self, 'fit_val'):
                fit_val = self.fit_val
                residual = np.divide((y - fit_val), sigma)
        elif fit_val is not None:
            residual = np.divide((y - fit_val), sigma)

        elif coeffs is True and x is True:
            if hasattr(self, 'coeffs') and hasattr(self, 'x'):
                coeffs = self.coeffs

                x = self.x

        elif coeffs is not None and x is not None:
            residual = np.divide((y - self.fit(coeffs, x)), sigma)

        else:
            sys.exit("{} has either no attribute 'coeffs' or 'x'"
                     .format(self) + " OR no attribute 'fit_val'.")

        return residual

    def residual_squared(self, x=None, y=None, sigma=None, coeffs=None,
                         fit_val=None, weighted=True):
        """
        Calculates either the weighted or unweighted squared
        residual

        """
        if y is True:
            if hasattr(self, 'y'):
                y = self.y
        elif y is None:
            sys.exit("{} has no attribute 'y'".format(self))

        if weighted is not True:
            sigma = 1
        elif sigma is True:
            if hasattr(self, 'sigma'):
                sigma = self.sigma
        elif sigma is None:
            sys.exit("{} has no attribute 'sigma'".format(self))

        if fit_val is True:
            if hasattr(self, 'fit_val'):
                fit_val = self.fit_val
                residual = np.divide((y - fit_val), sigma)
        elif fit_val is not None:
            residual = np.divide((y - fit_val), sigma)

        elif coeffs is True and x is True:
            if hasattr(self, 'coeffs') and hasattr(self, 'x'):
                coeffs = self.coeffs
                x = self.x
                residual = np.divide((y - self.fit(coeffs, x)), sigma)

        elif coeffs is not None and x is not None:
            residual = np.divide((y - self.fit(coeffs, x)), sigma)

        else:
            sys.exit("{} has either no attribute 'coeffs' or 'x'"
                     .format(self) + " OR no attribute 'fit_val'.")

        residual_squared = np.divide(np.square(residual), sigma)

        return residual_squared

    def chi_squared(self, x=None, y=None, sigma=None, coeffs=None,
                    fit_val=None):
        """
        Calculates the chi-squared over the data stored in the
        Residuals object

        The chi-squared is the sum of the weighted squared residuals.
                :math: sum_i( (y - y_)^2 / w)
        """

        residual_squared = self.residual_squared(x=x, y=y, sigma=sigma,
                                                 coeffs=coeffs,
                                                 fit_val=fit_val)

        return np.sum(residual_squared)

    def reduced_chi_squared(self, x=None, y=None, sigma=None, coeffs=None,
                            fit_val=None, n=None, m=None, v=None):
        """
        Calculates the reduced chi-squared over the data
        stored in the Residuals object.

        The reduced chi-squared is the chi-squared divided by
        the degrees of freedom.
                :math: chi-squared / v

        The degrees of freedom (v) are the number of observations (n)
        minus the number of fitted parameters (m).
                :math: v = n - m

        """

        # If nothing is provided
        if n is None and m is None and v is None:
            sys.exit("Reduced chi_squared expects either the number of " +
                     "observations and number of fitted parameters or " +
                     "the degrees of freedom. Currently none are given.")

        # If n and m are provided and v is not calculte v = n - m
        # otherwise use the v provided
        if n is not None and m is not None:
            if v is None:
                v = n - m

        # If n or m are not provided, then v must be provided
        if n is None or m is None:
            if v is None:
                sys.exit("Number of observations or number of fitted " +
                         "parameters and degrees of freedom are none.")

        chi_squared = self.chi_squared(x=x, y=y, sigma=sigma,
                                       coeffs=coeffs, fit_val=fit_val)

        return np.divide(chi_squared, v)

    def plot_residual(self, weighted=True, squared=False, hold=False):
        """
        Plots one of the following: weighted residual,
        unweighted residual, weighted residual squared,
        unweighted residual squared.
        """

        if squared:
            residuals = self.residual_squared(weighted)
            figure_str = "Residual Squared"

        else:
            residuals = self.residual(weighted)
            figure_str = "Residual"

        if weighted:
            figure_str = "Weighted " + figure_str
        else:
            figure_str = "Unweighted " + figure_str
            residuals = self.sigma * residuals

        plt.figure(figure_str)
        plt.title(figure_str)
        plt.scatter(self.x, residuals)
        plt.legend([figure_str])

        if not hold:
            plt.show()

    def strahl_fit(self, coeffs, x):
        # Extract (x,y) coordinates for knots of D and v
        Dx_knots = self.x_knots
        Dy_knots = coeffs[0:self.numb_knots]

        vx_knots = self.x_knots
        vy_knots = coeffs[self.numb_knots:self.numb_knots * 2]

        # Reset spline objects with new knots
        D_spline_attributes = self.D_spline_.re_init(Dx_knots, Dy_knots)
        v_spline_attributes = self.v_spline_.re_init(vx_knots, vy_knots)

        self.D_spline_, self.Dx_knots, self.Dy_knots = D_spline_attributes
        self.v_spline_, self.vx_knots, self.vy_knots = v_spline_attributes

        # Interpole over the splines
        D_profile = self.D_spline_(x)
        v_profile = self.v_spline_(x)

        # Create a quick input file
        inputs = [len(x), x, D_profile,
                  len(x), x, v_profile]

        strahl.quick_input_file(main_fn=self.main_fn, inpt_fn=self.inpt_fn,
                                inputs=inputs, verbose=self.verbose)

        # Run strahl
        strahl_cmd = None
        if self.strahl_verbose:
            strahl_cmd = "./strahl v"

        strahl.run(self.inpt_fn, strahl_cmd=strahl_cmd, verbose=self.verbose)

        # Extract emissivity
        variables = ['diag_lines_radiation']

        results = strahl.extract_results(result=self.data_fn,
                                         variables=variables)

        emissivity_ = results['variables']['diag_lines_radiation']

        # Scale factor for the profile
        scale = coeffs[4] * 50000

        # Extract profiles from emissivity_
        profiles = math.generate_profile(emissivity_,
                                         charge_index=self.charge_index,
                                         integrat_dim=self.integration_dim,
                                         signal_scale=scale)

        profile, scaled_profile = profiles

        return scaled_profile

    def re_init_splines(self, x=None, y=None, params=None):
        numb_knots = self.numb_knots

        if x is not None:
            if isinstance(x, (list, np.ndarray)):
                Dx_knots = x[0:numb_knots]
                vx_knots = x[numb_knots:2 * numb_knots]
            else:
                sys.exit("x must be an instance of list or np.ndarray.")

        if y is not None:
            if isinstance(y, (list, np.ndarray)):
                Dy_knots = y[0:numb_knots]
                vy_knots = y[numb_knots:2 * numb_knots]
            else:
                sys.exit("y must be an instance of list or np.ndarray.")

        D_reinit = self.D_spline_.re_init(Dx_knots, Dy_knots, params)
        v_reinit = self.v_spline_.re_init(vx_knots, vy_knots, params)

        self.D_spline_, self.Dx_knots, self.Dy_knots = D_reinit
        self.v_spline_, self.vx_knots, self.vy_knots = v_reinit

    @classmethod
    def strahl(cls, D_spline_, v_spline_, main_fn, inpt_fn, data_fn,
               charge_index, integration_dim,
               strahl_verbose=False, verbose=False):

        residual_ = cls(D_spline_, v_spline_, main_fn, inpt_fn, data_fn,
                        charge_index, integration_dim,
                        strahl_verbose=strahl_verbose, verbose=verbose)

        return residual_


class Splines:

    def __init__(self, spline_class, x, y, params):
        """
        Instantiates a spline object
        """

        self.spline_class = spline_class
        self.params = params

        self.spline = spline_class(x, y, **params)
        self.x_knots = self.spline.get_knots()
        self.y_knots = self.spline.get_coeffs()

    def __call__(self, x):
        """
        Interpolate with the spline attribute over the x values
        """
        return self.spline(x)

    def __str__(self):
        spline_dict = self.__dict__
        spline_str = "Splines Attributes:\n"

        for key, item in spline_dict.items():
            spline_str += "\t\t{}: {}\n".format(key, item)

        return spline_str



    def re_init(self, x=None, y=None, params=None):
        if x is not None or y is not None or params is not None:

            if x is not None:
                if not isinstance(x, (list, np.ndarray)):
                    sys.exit("Error: x not list or ndarray type.")
            else:
                x = self.x_knots

            if y is not None:
                if not isinstance(y, (list, np.ndarray)):
                    sys.exit("Error: y not list or ndarray type.")
            else:
                y = self.y_knots

            if len(x) != len(y):
                sys.exit("Error: length of x and y do not match")

            if params is not None:
                if isinstance(params, dict):
                    print("Using new value(s) for params.")
                    self.params = params
                else:
                    sys.exit("Error: parameters not dictionary type.")

            self.spline = self.spline_class(x, y, **self.params)
            self.x_knots = self.spline.get_knots()
            self.y_knots = self.spline.get_coeffs()

        return self, self.x_knots, self.y_knots

    @classmethod
    def univariate_spline(cls, x, y, order=1):
        """
        Class method to initialize a Splines object for an kth
        order univariate spline interpolation for k between
        1 <= k <= 5.
        """
        params = {'k': order}

        return cls(InterpolatedUnivariateSpline, x, y, params)

    @classmethod
    def linear_univariate_spline(cls, x, y):
        """
        Class method to initialize a Splines object for a linear
        univariate spline interpolation class.
        """

        return cls.univariate_spline(x, y, 1)

