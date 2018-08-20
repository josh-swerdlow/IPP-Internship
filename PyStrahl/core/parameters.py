########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Stores the parameter class which is used to order
information about parameters used to execute STRAHL.
"""

__author__ = 'Joshua Swerdow'

import os
import numpy


class Parameter():
    """
    The Parameter object encapsulates the state of a parameter
    and contains the methods to create every single type of parameter
    found within all of the parameter files (main, background, etc)
    """

    def __init__(self, name, val, priority,
                valid_vals=None, dtype=None, info=None, verbosity=False):
        """
        Initializes a Parameter object

        Args:
             name (str): name of the parameter
             priority (int): priority of the parameter within its'
                parameter file.
             val: value assigned to the parameter
             valid_vals (list): acceptable values for parameter
             dtype (list): acceptable data type classes for parameter
             info (str): A string that describes the parameter
                if full detail.
             verbosity (bool): turns on and off verbose output

        Attributes:
             name (str): name of the Parameter
             priority (int): priority of the Parameter within its'
                parameter file.
             value: value assigned to the parameter
             valid_vals (list): acceptable values for parameter
             dtype (list): acceptable data type classes for Parameter
             help (str): A string description of the Parameter
             verbose (bool): determines if verbose execution is used
             state (bool): determines whether we should put the
                value of the Parameter into the input file or leave it in
                the parameter file.

        .. :todo:
            Create to text outputs: verbose and debug
            Deal with priority
            Add integration of valid_vals and dtype
        """
        self.name = name
        self.priority = priority
        self.value = val

        if valid_vals is not None and len(valid_vals) == 0:
            print("valid_vals is empty. Accepting all values.")
            valid_vals = None
        self.valid_vals = valid_vals

        if dtype is not None and len(dtype) == 0:
            print("dtype is empty. Accepting all date types.")
            dtype = None
        self.dtype = dtype

        if info is None or not isinstance(info, str):
            info = ("There is no help doc for '{}'. Try viewing the manual."
                    .format(name))
        self.help = info

        if isinstance(verbosity, bool):
            self.verbose = verbosity

            if self.verbose:
                print(">>> Initialized a Parameter object for '{}' with attributes:"
                      .format(self.name))
                print(self.__dict__)

        else:
            self.verbose = False

        self.state = False

    def attributes(self):
        """
        Prints out all of the attributes of the Parameter object
        """
        if self.verbose:
            print("Printing attributes ...\n")

        print("Attributes of {} object {}:".format(self.__class__, self))

        print(self.__dict__)

    def attribute_dictionary(self, keys=None):
        """
        Generates a dictionary of the attributes of the
        Parameter object.

        Args:
             keys (list): A list of attribute names from the
                Parameter object that you would like to have in the
                dictionary.

        Returns:
             A Dictionary of all attriutes requested by the user from
             the Parameter object.
        """
        if self.verbose:
            print("Generating a dictionary of {}'s attributes ..."
                .format(self))

        attributes = dict()

        if keys is None:
            keys = self.__dict__.keys()

        for key in keys:
            val = self.__dict__[key]

            if isinstance(val, numpy.ndarray):
                val = val.tolist()

            attributes[key] = val

        return attributes

    def change_value(self, val=None, verbose=False):
        """
        Changes the value of the parameter to val or, if not given,
        whatever the user enters. The value of the parameter must be
        compatible with the parameter objects dtype attribute.

        Args:
             val: The new value of the parameter
             verbose (bool): Turns on and off verbose output

        Attributes:
             val: The new value of the parameter
        """

        if val is None:
            val = query.evaluate()
            self.changeVal(val)

        if self.verbose or verbose:
            print("Changing value of '{}' to {}".format(self.name, val))

        self.value = val

    def change_state(self, state=None):
        """
        Changes the state of the Parameter object to either the
        opposite of what it currently is or to what the user passes.

        Args:
            state (bool): Determine if we need to enter the value
                into an input file or not
        """
        if self.verbose:
            print("Changing state of {} ... ".format(self.name))

        if state is None:
            state = not self.state

        if isinstance(state, bool):
            self.state = state

        else:
            print('state must be a boolean variable')
            raise TypeError

        if self.verbose:
            print("Changed the state of {} to {}"
                  .format(self.name, self.state))

# Main Input Parameters -- at the moment this is limited
    @classmethod
    def atomic_weight(cls, value, verbosity=False):
        """Method to generate the atomic weight parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "atomic weight"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 1, info=info, verbosity=verbosity)

    @classmethod
    def charge(cls, value, verbosity=False):
        """Method to generate the charge parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "charge"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 2, info=info, verbosity=verbosity)

    @classmethod
    def shot(cls, value, verbosity=False):
        """Method to generate the shot parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "shot"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 3, info=info, verbosity=verbosity)

    @classmethod
    def index(cls, value, verbosity=False):
        """Method to generate the index parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "index"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 4, info=info, verbosity=verbosity)

    @classmethod
    def rho(cls, value, verbosity=False):
        """Method to generate the rho parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "rho"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 5, info=info, verbosity=verbosity)

    @classmethod
    def number_grid_points(cls, value, verbosity=False):
        """Method to generate the number of grid points parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "number of grid points"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 6, info=info, verbosity=verbosity)

    @classmethod
    def dr_0(cls, value, verbosity=False):
        """Method to generate the dr_0 parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "dr_0"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 7, info=info, verbosity=verbosity)

    @classmethod
    def dr_1(cls, value, verbosity=False):
        """Method to generate the dr_1 parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "dr_1"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 8, info=info, verbosity=verbosity)

    @classmethod
    def number_of_changes(cls, value, verbosity=False):
        """Method to generate the number of changes parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "number of changes (start-time+... +stop-time)"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 9, info=info, verbosity=verbosity)

    @classmethod
    def time(cls, value, verbosity=False):
        """Method to generate the time parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "time"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 10, info=info, verbosity=verbosity)

    @classmethod
    def dt_start(cls, value, verbosity=False):
        """Method to generate the dt at start parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "dt at start"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 11, info=info, verbosity=verbosity)

    @classmethod
    def dt_increase(cls, value, verbosity=False):
        """Method to generate the increase of dt after cycle parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "increase of dt after cycle"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 12, info=info, verbosity=verbosity)

    @classmethod
    def steps_per_cycle(cls, value, verbosity=False):
        """Method to generate the steps per cycle parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "steps per cycle"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 13, info=info, verbosity=verbosity)

    @classmethod
    def number_of_impurities(cls, value, verbosity=False):
        """Method to generate the number of impurities parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "number of impurities"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 14, info=info, verbosity=verbosity)

    @classmethod
    def element(cls, value, verbosity=False):
        """Method to generate the element parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "element"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 15, info=info, verbosity=verbosity)

    @classmethod
    def impure_atomic_weight(cls, value, verbosity=False):
        """Method to generate the impure atomic weight parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "impure atomic weight"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 16, info=info, verbosity=verbosity)

    @classmethod
    def energy_of_neutrals(cls, value, verbosity=False):
        """Method to generate the energy of neutrals parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "energy of neutrals(eV)"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 17, info=info, verbosity=verbosity)

    @classmethod
    def diffusion_num_intpln_pts(cls, value, verbosity=False):
        """Method to generate the profile"""
        name = "# of D interpolation points"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 18, info=info, verbosity=verbosity)

    @classmethod
    def diffusion_rho_pol_grid(cls, value, verbosity=False):
        """Method to generate the profile"""
        name = "rho polodial grid for D interpolation"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 19, info=info, verbosity=verbosity)

    @classmethod
    def diffusion(cls, value, verbosity=False):
        """Method to generate the diffusion profile"""
        name = "D[m^2/s]"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 20, info=info, verbosity=verbosity)

    @classmethod
    def conv_veloc_num_intpln_pts(cls, value, verbosity=False):
        """Method to generate the profile"""
        name = "# of v interpolation points"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 21, info=info, verbosity=verbosity)

    @classmethod
    def conv_veloc_rho_pol_grid(cls, value, verbosity=False):
        """Method to generate the profile"""
        name = "rho polodial grid for v interpolation"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 22, info=info, verbosity=verbosity)

    @classmethod
    def convective_velocity(cls, value, verbosity=False):
        """Method to generate the diffusion profile"""
        name = "v[m/s]"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, 23, info=info, verbosity=verbosity)

# Background Input Parameters
    @classmethod
    def ne_numTimePts(cls, value, verbosity=False):
        """
        Method to generate the electon density number of time
        points parameter
        """
        name = "ne_numTimePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_timePts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_timePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_paramType(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_paramType"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_radCoord(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_radCoord"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_numInterpPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_numInterpPts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_radGrid(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_radGrid"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_radGridPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_radGridPts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ne_decayLength(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ne_decayLength"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_numTimePts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_numTimePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_timePts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_timePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_paramType(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_paramType"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_radCoord(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_radCoord"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_numInterpPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_numInterpPts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_radGrid(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_radGrid"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_radGridPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_radGrid"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def te_decayLength(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "te_decayLength"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_numTimePts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_numTimePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_timePts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_timePts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_paramType(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_paramType"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name9, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_radCoord(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_radCoord"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name0, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_numInterpPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_numInterpPts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_radGrid(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_radGrid"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_radGridPts(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_radGridPts"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    @classmethod
    def ti_decayLength(cls, value, verbosity=False):
        """Method to generate the  parameter

        Args:
            value: The value to be assigned to Parameter object
                upon initialization.
            verbosity (bool): Turns on or off verbose execution
                for all methods for this object.

        Returns:
            A Parameter object
        """
        name = "ti_decayLength"
        info = Parameter._help(name)
        dtype = None
        valid_vals = None

        return cls(name, value, info=info, verbosity=verbosity)

    # Geometry Input Parameters

    # Flux Input Parameters

    @staticmethod
    def _help(name):
        """Finds the help file name.h.

        Parameters:
            name (str): Name of the help file with or without '.h'

        Returns:
            A string that describes the parameter, valid data type(s),
            and valid values.
        """

        if not name.endswith(".h"):
            name = name + ".h"

        file = os.path.join("./source/interface", "help", name)

        help_str = None
        if os.path.isfile(file):
            with open(file, "r") as f:
                help_str = f.read()

        return help_str


class Parameters():
    """
    Class to organize multiple parameters.
    """

    def __init__(self, name, parameters, verbosity):
        self.parameter_dict = dict()
        for parameter in parameters:
            self.parameter_dict[parameter.name] = parameter

        self.name = name

        self.rows = 1
        self.cols = len(parameters)

        self.state = False

        self.verbose = verbosity
        if self.verbose:
            print(">>>>>Initialized a Parameters object called {} with attributes:\n{}"
                .format(self.name, self.parameter_dict))

    @classmethod
    def impurities(cls, parameters, verbosity=False):
        name = "impurities"

        return cls(name, parameters, verbosity)

    @classmethod
    def main_time_steps(cls, parameters, verbosity=False):
        name = "main_time_steps"

        return cls(name, parameters, verbosity)









