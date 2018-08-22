import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from PyStrahl.core import strahl
from PyStrahl.utils import math
from scipy.interpolate import InterpolatedUnivariateSpline


class Function:
    """Create a Function object that stores certain features of the
    given function; such as, the given free parameters. In addition,
    one can evaluate the function over a range and add noise sampled
    from given normal distribution.

    """
    rndst = np.random.RandomState(0)

    def __init__(self, name, math, func,
                 parameterization=None, eqtn=None, data=None,
                 **free_params):
        """Initializes the Function object

        Attributes:
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
        """Creates a linear Function object"""
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
        """Creates a quadratic Function object"""
        print("Creating a quadratic function...")

        name = "quadratic"
        if a != 0:
            name = "2nd order quadratic"
            if parameterization is not None:
                name = name + " with parameterization"

        elif b != 0:
            name = "1st order quadratic"
            if parameterization is not None:
                name = name + " with parameterization"

        else:
            name = "0th order quadratic"
            if parameterization is not None:
                name = name + " with parameterization"

        math = "ax^2 + bx + c"

        return cls(name, math, Function._quadFunc,
                   parameterization=parameterization,
                   a=a, b=b, c=c)

    @classmethod
    def constant(cls, a):
        """Creates a constant Function object"""
        print("Creating a constant function...")

        if a is None:
            a = 0

        name = "constant"
        math = "a"

        return cls(name, math, Function._constFunc, a=a)

    @classmethod
    def noise(cls, sigma_background, scale_factor, parameterization=None):
        """Creates a noise Function object"""
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
        """A linear function"""
        return m * x + b

    @staticmethod
    def _quadFunc(x, a, b, c):
        """A quadratic function"""
        return a * x**2 + b * x + c

    @staticmethod
    def _constFunc(a):
        """A constant function"""
        return a

    @staticmethod
    def _noiseFunc(intensity, sig, A):
        """A noise function that scales with some intensity function"""
        return (np.random.normal(0, sig, len(intensity)) +
                np.random.normal(0, np.sqrt(A * intensity), len(intensity)))


class Residuals:

    def __init__(self, spline_, x=None, y=None, sigma=None, verbose=False):
        """
        Initializes a Residuals object which can calculate
        various parameters (weighted_residual, unweight_residual,
        residual, chi_squared, and reduced chi_squared)
        """
        self.spline_ = spline_

        if x is None:
            print("x is NoneType. Use Residuals.set() to assign a value.")
        else:
            self.x = x

        if y is None:
            print("y is NoneType. Use Residuals.set() to assign a value.")
        else:
            self.y = y

        if sigma is None:
            print("sigma is NoneType. Use Residuals.set() to assign a value.")
        else:
            self.sigma = sigma

        self.verbose = verbose
        if self.verbose:
            print("Executing with verbose output.")

    def __call__(self, coeffs=None, x=None, y=None, sigma=None):

        if coeffs is not None:
            self.re_init_spline(y=coeffs)

        return self.residual(x=x, y=y, sigma=sigma)

    def __str__(self):
        residual_str = ("residual attributes: {}\n\nspline attributes: {}"
                        .format(self.__dict__, self.spline_.__dict__))

        return str(residual_str)

    def re_init_spline(self, x=None, y=None, params=None):
        self.spline_ = self.spline_.re_init(x=x, y=y, params=params)

    def set(self, spline_=None, x=None, y=None, sigma=None):
        if spline_ is not None:
            print("Assigned spline_ to {}".format(spline_))
            self.spline_ = spline_

        if x is not None:
            print("Assigned x to {}".format(x))
            self.x = x

        if y is not None:
            print("Assigned y to {}".format(y))
            self.y = y

        if sigma is not None:
            print("Assigned sigma to {}".format(sigma))
            self.sigma = sigma

    def residual(self, x=None, y=None, sigma=None, weighted=True):
        """
        Calculates either the weighted or unweighted residual
        of the data stored in the Residuals object

        Notes:

        The function executes in the following way depending and
        the parameters given.
        For x, if:
            x is a param     and x is an attribute     --> use param
            x is a param     and x is not an attribute --> use param
            x is not a param and x is an attribute     --> use attribute
            x is not a param and x is not an attribute --> exit
        """

        if x is not None or hasattr(self, 'x'):
            if x is None:
                if self.verbose:
                    print("Using the attribute x.")

                x = self.x

            else:
                if self.verbose:
                    print("Using parameter x.")

        else:
            sys.exit("Residual requires a x value.")

        if y is not None or hasattr(self, 'y'):
            if y is None:
                if self.verbose:
                    print("Using the attribute y.")

                y = self.y

            else:
                if self.verbose:
                    print("Using parameter y.")

        else:
            sys.exit("Residual requires a y value.")

        if sigma is not None or hasattr(self, 'sigma'):
            if sigma is None:
                if self.verbose:
                    print("Using the attribute sigma.")

                sigma = self.sigma

            else:
                if self.verbose:
                    print("Using parameter sigma.")

        elif not weighted:
            if self.verbose:
                print("Calculating unweighted residual, sigma not needed.")

            sigma = 1

        else:
            sys.exit("Residual requires a sigma value.")

        return np.divide((y - self.spline_(x)), sigma)

    def residual_squared(self, weighted=True):
        """
        Calculates either the weighted or unweighted squared
        residual of the data stored in the Residuals object
        """

        if not weighted:
            sigma = 1
        else:
            sigma = self.sigma

        return np.square(self.y - self.spline_(self.x)) / sigma

    def chi_squared(self):
        """
        Calculates the chi-squared over the data stored in the
        Residuals object

        The chi-squared is the sum of the weighted squared residuals.
                :math: sum_i( (y - y_)^2 / w)
        """

        return np.sum(self.residual_squared(weighted=True))

    def reduced_chi_squared(self, n=None, m=None, v=None):
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

        return (self.chi_squared() / v)

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

        inpt_path = os.path.join("./", inpt_fn)
        if not os.path.isfile(inpt_path):
            sys.exit("{} is not a valid input file.".format(inpt_fn))

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
            fit=None, x=None, y=None, sigma=None):

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

    def mpfit_residual(self, coeffs, x, y, sigma):
        fit = self.fit(coeffs=coeffs, x=x)

        residual = np.divide(y - fit, sigma)

        return {'residual': residual}

    def residual(self, x=None, y=None, sigma=None, weighted=True):
        """
        Calculates either the weighted or unweighted residual
        of the data stored in the Residuals object

        Notes:

        The function executes in the following way depending and
        the parameters given.
        For x, if:
            x is a param     and x is an attribute     --> use param
            x is a param     and x is not an attribute --> use param
            x is not a param and x is an attribute     --> use attribute
            x is not a param and x is not an attribute --> exit
        """

        if x is not None or hasattr(self, 'x'):
            if x is None:
                if self.verbose:
                    print("Using the attribute x.")

                x = self.x

            else:
                if self.verbose:
                    print("Using parameter x.")

        else:
            sys.exit("Residual requires a x value.")

        if y is not None or hasattr(self, 'y'):
            if y is None:
                if self.verbose:
                    print("Using the attribute y.")

                y = self.y

            else:
                if self.verbose:
                    print("Using parameter y.")

        else:
            sys.exit("Residual requires a y value.")

        if sigma is not None or hasattr(self, 'sigma'):
            if sigma is None:
                if self.verbose:
                    print("Using the attribute sigma.")

                sigma = self.sigma

            else:
                if self.verbose:
                    print("Using parameter sigma.")

        elif not weighted:
            if self.verbose:
                print("Calculating unweighted residual, sigma not needed.")

            sigma = 1

        else:
            sys.exit("Residual requires a sigma value.")

        return np.divide((y - self.spline_(x)), sigma)

    def residual_squared(self, weighted=True):
        """
        Calculates either the weighted or unweighted squared
        residual of the data stored in the Residuals object
        """

        if not weighted:
            sigma = 1
        else:
            sigma = self.sigma

        return np.square(self.y - self.fit(self.x)) / sigma

    def chi_squared(self):
        """
        Calculates the chi-squared over the data stored in the
        Residuals object

        The chi-squared is the sum of the weighted squared residuals.
                :math: sum_i( (y - y_)^2 / w)
        """

        return np.sum(self.residual_squared(weighted=True))

    def reduced_chi_squared(self, n=None, m=None, v=None):
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

        return (self.chi_squared() / v)

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

        return profile

    @classmethod
    def strahl(cls, D_spline_, v_spline_, main_fn, inpt_fn, data_fn,
               strahl_verbose=False, verbose=False):

        residual_ = cls(D_spline_, v_spline_, main_fn, inpt_fn, data_fn,
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

