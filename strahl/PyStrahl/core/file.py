########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: Stores the parameter, input, and summary file classes
    which are used to order information and methods that
    create, interact, or load the parameter, input, or summary
    files.
"""

__author__ = 'Joshua Swerdow'

import os
import re
import sys

import subprocess as sub
import numpy as np

from PyStrahl.core.parameters import Parameter
from PyStrahl.utils.auxillary import color_file, color_dir, print_directory


class ParameterFile():
    """
    This object ecapsulates the entire state of a given parameter file.

    A parameter file is defined as a main, background, geometry, flux,
    or atomic data file which has the parameter values for STRAHL to read.

    Each file has a different file name syntax and storage location. Please
    see the STRAHL manual for more information.
        * ./param: main files, atomic data
        * ./nete: background, geometry, flux
    """

    def __init__(self, param_type, fn, priority,
                 params=None, groups=None,
                 verbosity=False):
        """
        Initiatlizes a ParameterFile object.

        Args:
            param_type (str): The type of parameter file which is being created
                i.e) 'main', 'background', 'geometry', 'flux', 'atomic_data'
            fn (str): The file's name
            priority (int): The priority of the parameter file in the
                input file.
            params (dict): Dictionary of initialized Parameter objects
                with associated names.
            groups (list): Nested lists of parameter object names which are
                found together on a single line.
            verbosity (bool): Turns on and off verbose ouput

        Attributes:
            type (str): 'main', 'background', 'geometry', 'flux', 'atomic_data'
            fn (str): The files name
            priority (int): The priority of the parameter file in the
                input file.
            parameter_dict (dict): Dictionary of initialized parameter objects
                with associated names.
            size (int): Size of the parameter_dict
            groups (list): Allows code to search for groups of parameters
                rather than just single parameters.
                For more detail, see manual.
            verbose (bool): Determines if verbose execution is used
        """

        self.type = param_type
        self.fn = fn
        self.priority = priority

        if params is None:
            sys.exit("Error: params is NoneType.")
        self.parameter_dict = params

        self.size = len(params)

        if groups is None:
            sys.exit("groups is NoneType.")
        self.groups = groups

        if isinstance(verbosity, bool):
            self.verbose = verbosity

            if self.verbose:
                print(">>> Initialized a ParameterFile object with attributes:")
                print(self.attributes(True))
        else:
            self.verbose = False

    def attributes(self, all=False):
        """
        Prints out all of the attributes of the parmeterFile object

        Args:
            all (bool): Determines whether to recursively call
                parameters.attributes() on each parameter object in
                paramaterFile.parameter_dict

        """
        if self.verbose:
            print("Printing attributes ...\n")

        print("Attributes of {} object {}:".format(self.__class__, self))

        print(self.__dict__)

        # Iteratively calls the attributes method on parameters
        if all:
            [param.attributes() for param in self.parameter_dict.values()
                if isinstance(param, Parameter)]

    def reset_states(self):
        """
        Iteratively resets all the states for each Parameter object
        found in the ParameterFile's parameter_dict attribute.
        """

        if self.verbose:
            print("Resetting states ...\n")

        for param in self.parameter_dict.values():
            param.change_state(False)

    def attribute_dictionary(self, keys=None, param_keys=None, dic=None):
        """
        Recursively generates nested dictionaries of the attributes of the
        ParameterFile object.

        Args:
            keys (list): A list of attribute names from the
                ParameterFile object that you would like to have in the
                dictionary. If not given, it will assume all of the keys
                of the dic object.
            param_keys (list): A list of attribute names from
                the parameter object that you would like to have in the
                dictionary. If not given

            dic (dict): A dictionary object to operate on (presumably a
                ParameterFile's params_dict). If not given, it will assume
                the object __dict__ is the appropriate dictionary.

        Returns:
            A Nested dictionary of all attributes requested by the user
            from the ParameterFile object and the parameters objects
            within. Note: This will turn any np.ndarray instance into a
            standard Python list instances.
        """
        if dic is None or not isinstance(dic, dict):
            obj = self
            dic = self.__dict__
        else:
            obj = dic

        if self.verbose:
            print("Generating a dictionary of {}'s attributes ..."
                .format(obj))

        attributes = dict()

        if keys is None or not isinstance(keys, list):
            keys = dic.keys()

        if self.verbose:
            print("Grabbing the following attributes:")
            print(keys)

        for key in keys:
            val = dic[key]

            if isinstance(val, dict):
                keys = list(val.keys())
                val = self.attribute_dictionary(keys, param_keys, val)

            if isinstance(val, Parameter):
                val = val.attribute_dictionary(param_keys)

            if isinstance(val, np.ndarray):
                val = val.tolist()

            attributes[key] = val

        return attributes

    def change_value(self, param, val):
        """
        Changes the value for the parameters object to val using the
        parameters.change_val() method.

        Args:
            param (:obj:`Parameter`): A Parameter object to operate on
            val: A value(s) to be assigned to the param Parameter object.
        """
        if param is None:
            sys.exit("Error: param is None.")

        param.change_value(val, self.verbose)

    def in_group(self, parameter):
        key_group = None

        for group in self.groups:
            if parameter in group:
                key_group = group

        return key_group

    @classmethod
    def create_parameter_files(cls,
                       main_fn=None,
                       bckg_fn=None,
                       geom_fn=None,
                       flux_fn=None,
                       verbosity=False):
        """Creates multiple parameter file objects depending on the file_names.

        Args:
            main_fn (str): A main file name
            bckg_fn (str): A background file name
            geom_fn (str): A geometry file name
            flux_fn (str): A flux file name
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A list of ParameterFile objects for the file names given in order
            of main, bckg, geom, flux. Does not return anything if a file name
            for the parameter file type is not given. Returns an empty list if
            no file names are given.
        """
        main_dir = "param_files"
        bckg_dir = "nete"
        geom_dir = bckg_dir
        flux_dir = bckg_dir

        main = None
        bckg = None
        geom = None
        flux = None

        parameter_files = list()

        if main_fn is not None:
            if os.path.dirname(main_fn) is not '':
                main_fn = os.path.basename(main_fn)

            main_path = os.path.join(main_dir, main_fn)

            if os.path.isfile(main_path):

                main = cls._create_parameter_file(main_fn,
                                            main_path,
                                            "main",
                                            verbosity)

                parameter_files.append(main)

        if bckg_fn is not None:
            if os.path.dirname(bckg_fn) is not '':
                bckg_fn = os.path.basename(bckg_fn)

            bckg_path = os.path.join(bckg_dir, bckg_fn)

            if os.path.isfile(bckg_path):

                bckg = cls._create_parameter_file(bckg_fn,
                                            bckg_path,
                                            "background",
                                            verbosity)

                parameter_files.append(bckg)

        if geom_fn is not None:
            if os.path.dirname(geom_fn) is not '':
                geom_fn = os.path.basename(geom_fn)

            geom_path = os.path.join(geom_dir, geom_fn)

            if os.path.isfile(geom_path):

                geom = cls._create_parameter_file(geom_fn,
                                            geom_path,
                                            "geometry",
                                            verbosity)

                parameter_files.append(geom)

        if flux_fn is not None:
            if os.path.dirname(flux_fn) is not '':
                flux_fn = os.path.basename(flux_fn)

            flux_path = os.path.join(flux_dir, flux_fn)

            if os.path.isfile(flux_path):

                flux = cls._create_parameter_file(flux_fn,
                                            flux_path,
                                            "flux",
                                            verbosity)

                parameter_files.append(flux)

        return parameter_files

    @classmethod
    def create_parameter_file(cls, fn, file_path, file_type, verbosity):
        """Creates a single ParameterFile object depending on the file_type.

        If the file path given is not a valid file, then it prompts the user
        for a new file name. In addition, it gives the user an ouput of valid
        files from the list of files found in the appropriate directory for
        that parameter file type.

        Parameters:
            fn (str): The name of the file
            file_path (str): The path to the file
            file_type (str): The type of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A ParameterFile object.
        """
        parameter_file = None

        if os.path.isfile(file_path):
            if file_type is "main":
                parameter_file = cls._main_file(fn, verbosity)
            elif file_type is "background":
                parameter_file = cls._background_file(fn, verbosity)
            elif file_type is "geometry":
                pass
            elif file_type is "flux":
                pass

        else:
            file = color_file(fn)
            path = color_dir(file_path)

            print("Could not find {} in {}!\n".format(file, path))

            print_directory(file_path)

            new_fn = input("Please enter a new file or exit [enter]: ")

            if new_fn is not '':
                parameter_file = cls.create_parameter_file(new_fn,
                                                    file_path,
                                                    file_type,
                                                    verbosity)

        return parameter_file

    @classmethod
    def _main_file(cls, fn, verbosity):
        """Class creation method for a main file ParameterFile object.

        This classmethod abstracts away a lot of the arguments needed
        to create a ParameterFile object because it is specifically
        designed to create main file type ParameterFile objects.

        Parameters:
            fn [str]: name of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A main file type ParameterFile object.
        """

        params, groups = ParameterFile._empty_main(verbosity)

        return cls("main", fn, 1, params, groups, verbosity)

    @classmethod
    def _background_file(cls, fn, verbosity):
        """
        Class creation method for a background file ParameterFile object.

        This classmethod abstracts away a lot of the arguments needed
        to create a ParameterFile object because it is specifically
        designed to create background file type ParameterFile objects.

        Parameters:
            fn [str]: name of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A background file type ParameterFile object.
        """

        params, groups = ParameterFile._empty_background(verbosity)

        return cls("background", fn, 2, params, groups, verbosity)

    @classmethod
    def _geometry_file(cls, fn, verbosity):
        """Class creation method for a geometry file ParameterFile object.

        This classmethod abstracts away a lot of the arguments needed
        to create a ParameterFile object because it is specifically
        designed to create geometry file type ParameterFile objects.

        Parameters:
            fn [str]: name of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A geometry file type ParameterFile object.
        """
        params, groups = ParameterFile._empty_geometry(verbosity)

        return cls("geometry", fn, 3, params, groups, verbosity)

    @classmethod
    def _flux_file(cls, fn, verbosity):
        """Class creation method for a flux file ParameterFile object.

        This classmethod abstracts away a lot of the arguments needed
        to create a ParameterFile object because it is specifically
        designed to create flux file type ParameterFile objects.

        Parameters:
            fn [str]: name of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A flux file type ParameterFile object.
        """
        params, groups = ParameterFile._empty_flux(verbosity)

        return cls("flux", fn, 4, params, groups, verbosity)

    @classmethod
    def _atom_file(cls, fn, verbosity):
        """Class creation method for a atom file ParameterFile object.

        This classmethod abstracts away a lot of the arguments needed
        to create a ParameterFile object because it is specifically
        designed to create atom file type ParameterFile objects.

        Parameters:
            fn [str]: name of parameter file
            verbosity (bool): Turns on and off verbose ouput

        Returns:
            A atom file type ParameterFile object.
        """

        params, groups = ParameterFile._empty_atomic_data(verbosity)

        return cls("atomic data", fn, 2, params, groups, verbosity)

    @staticmethod
    def _empty_main(verbosity=False):
        """Generates parameters dictionary and group for the main ParameterFile object.

        Initiates every Parameter object that can be found
        within a main parameter file by using the classmethods defined
        within ./parameters.py. Then organizes them into a dictionary.
        Additionally, this creates the nested list of groups required by
        ParameterFile objects for the main file.

        Args:
            verbosity (bool): Turns on and off verbose ouput for all
                Parameter objects

        Returns:
            A nested list of groups of main file Parameter object names, and a
            dictionary of main file Parameter objects.
        """
        ZERO = np.array([0], dtype=np.float)

        # M A I N  I O N
        atomic_weight = Parameter.atomic_weight(ZERO, verbosity)
        charge = Parameter.charge(ZERO, verbosity)

        # G R I D - F I L E
        shot = Parameter.shot(ZERO, verbosity)
        index = Parameter.index(ZERO, verbosity)

        # G R I D  P O I N T S  A N D  I T E R A T I O N
        rho = Parameter.rho(ZERO, verbosity)
        number_grid_points = Parameter.number_grid_points(ZERO, verbosity)
        dr_0 = Parameter.dr_0(ZERO, verbosity)
        dr_1 = Parameter.dr_1(ZERO, verbosity)

        # T I M E  S T E P S
        number_of_changes = Parameter.number_of_changes(ZERO, verbosity)
        time = Parameter.time(ZERO, verbosity)
        dt_start = Parameter.dt_start(ZERO, verbosity)
        dt_increase = Parameter.dt_increase(ZERO, verbosity)
        steps_per_cycle = Parameter.steps_per_cycle(ZERO, verbosity)

        # S T A R T  I M P U R I T Y  E L M E M E N T S
        number_of_impurities = Parameter.number_of_impurities(ZERO, verbosity)
        element = Parameter.element(ZERO, verbosity)
        impure_atomic_weight = Parameter.impure_atomic_weight(ZERO, verbosity)
        energy_of_neutrals = Parameter.energy_of_neutrals(ZERO, verbosity)

        # A N O M A L O U S  T R A N S P O R T
        d_num_intpln_pts = Parameter.diffusion_num_intpln_pts(ZERO, verbosity)
        d_rho_pol_grid = Parameter.diffusion_rho_pol_grid(ZERO, verbosity)
        diffusion = Parameter.diffusion(ZERO, verbosity)
        v_num_intpln_pts = Parameter.conv_veloc_num_intpln_pts(ZERO, verbosity)
        v_rho_pol_grid = Parameter.conv_veloc_rho_pol_grid(ZERO, verbosity)
        convective_velocity = Parameter.convective_velocity(ZERO, verbosity)

        parameter_dict = {
            atomic_weight.name: atomic_weight,
            charge.name: charge,
            shot.name: shot,
            index.name: index,
            rho.name: rho,
            number_grid_points.name: number_grid_points,
            dr_0.name: dr_0,
            dr_1.name: dr_1,
            number_of_changes.name: number_of_changes,
            time.name: time,
            dt_start.name: dt_start,
            dt_increase.name: dt_increase,
            steps_per_cycle.name: steps_per_cycle,
            number_of_impurities.name: number_of_impurities,
            element.name: element,
            impure_atomic_weight.name: impure_atomic_weight,
            energy_of_neutrals.name: energy_of_neutrals,
            d_num_intpln_pts.name: d_num_intpln_pts,
            d_rho_pol_grid.name: d_rho_pol_grid,
            diffusion.name: diffusion,
            v_num_intpln_pts.name: v_num_intpln_pts,
            v_rho_pol_grid.name: v_rho_pol_grid,
            convective_velocity.name: convective_velocity
        }

        groups = [[element.name, impure_atomic_weight.name,
                   energy_of_neutrals.name],

                  [time.name, dt_start.name, dt_increase.name,
                   steps_per_cycle.name]]

        return parameter_dict, groups

    @staticmethod
    def _empty_background(verbosity):
        """
        Generates parameters dictionary and group for the background
        ParameterFile object.

        Initiates every Parameter object that can be found
        within a background parameter file by using the classmethods defined
        within ./parameters.py. Then organizes them into a dictionary.
        Additionally, this creates the nested list of groups required by
        ParameterFile objects for the background file.

        Args:
            verbosity (bool): Turns on and off verbose ouput for all
                Parameter objects

        Returns:
            A nested list of groups of background file Parameter object names,
            and a dictionary of background file Parameter objects.
        """
        ZERO = np.array([0], dtype=np.float)

        parameter_dict = {
            # "time-vector": Parameter.ne_numTimePts(ZERO, verbosity),
            # "time-vector": Parameter.ne_timePts(ZERO, verbosity),
            # "time-vector": Parameter.te_numTimePts(ZERO, verbosity),
            # "time-vector": Parameter.te_timePts(ZERO, verbosity),
            # "time-vector": Parameter.ti_numTimePts(ZERO, verbosity),
            # "time-vector": Parameter.ti_timePts(ZERO, verbosity)
        }

        groups = []

        # parameter_dict = (Parameter.ne_numTimePts(ZERO),
        #             Parameter.ne_timePts(ZERO),
        #             Parameter.ne_paramType(ZERO),
        #             Parameter.ne_radCoord(ZERO),
        #             Parameter.ne_numInterpPts(ZERO),
        #             Parameter.ne_radGrid(ZERO),
        #             Parameter.ne_radGridPts(ZERO),
        #             Parameter.ne_decayLength(ZERO),
        #             Parameter.te_numTimePts(ZERO),
        #             Parameter.te_timePts(ZERO),
        #             Parameter.te_paramType(ZERO),
        #             Parameter.te_radCoord(ZERO),
        #             Parameter.te_numInterpPts(ZERO),
        #             Parameter.te_radGrid(ZERO),
        #             Parameter.te_radGridPts(ZERO),
        #             Parameter.te_decayLength(ZERO),
        #             Parameter.ti_numTimePts(ZERO),
        #             Parameter.ti_timePts(ZERO),
        #             Parameter.ti_paramType(ZERO),
        #             Parameter.ti_radCoord(ZERO),
        #             Parameter.ti_numInterpPts(ZERO),
        #             Parameter.ti_radGrid(ZERO),
        #             Parameter.ti_radGridPts(ZERO),
        #             Parameter.ti_decayLength(ZERO))

        return parameter_dict, groups

    @staticmethod
    def _empty_geometry(verbosity):
        """
        Generates parameters dictionary and group for the geometry
        ParameterFile object.

        Initiates every Parameter object that can be found
        within a geometry parameter file by using the classmethods defined
        within ./parameters.py. Then organizes them into a dictionary.
        Additionally, this creates the nested list of groups required by
        ParameterFile objects for the geometry file.

        Args:
            verbosity (bool): Turns on and off verbose ouput for all
                Parameter objects

        Returns:
            A nested list of groups of geometry file Parameter object names,
            and a dictionary of geometry file Parameter objects.
        """
        pass

    @staticmethod
    def _empty_flux(verbosity):
        """
        Generates parameters dictionary and group for the flux
        ParameterFile object.

        Initiates every Parameter object that can be found
        within a flux parameter file by using the classmethods defined
        within ./parameters.py. Then organizes them into a dictionary.
        Additionally, this creates the nested list of groups required by
        ParameterFile objects for the flux file.

        Args:
            verbosity (bool): Turns on and off verbose ouput for all
                Parameter objects

        Returns:
            A nested list of groups of flux file Parameter object names,
            and a dictionary of flux file Parameter objects.
        """
        pass

    @staticmethod
    def _empty_atomic_data(verbosity):
        """
        Generates parameters dictionary and group for the atomic data
        ParameterFile object.

        Initiates every Parameter object that can be found
        within a atomic data parameter file by using the classmethods defined
        within ./parameters.py. Then organizes them into a dictionary.
        Additionally, this creates the nested list of groups required by
        ParameterFile objects for the atomic data file.

        Args:
            verbosity (bool): Turns on and off verbose ouput for all
                Parameter objects

        Returns:
            A nested list of groups of atomic data file Parameter object names,
            and a dictionary of atomic data file Parameter objects.
        """
        pass


class InputFile():
    """
    This object ecapsulates the entire state of an input file.

    A input file is defined as a list of inputs that STRAHL
    expects to read in from the parameter files, but instead we
    have chosen to give them through an input file.

    Input files are store in the same directory the STRAHL is executed from.
    """

    def __init__(self, inpt_fn, inputs=None,
                 overwrite_flag=False, verbosity=False):
        """Initializes an InputFile object.

        Args:
            inpt_fn (str): The name of the input file
            inputs (list: The list of values of the parameters.
            overwrite_flag (bool): Determines if the  given input file
                will be automatically overwritten. If not boolean, then
                the default is False.
            verbosity (bool): Turns on and off verbose ouput

        Attributes:
            inputs (list): List of values of the parameters
            overwrite_flag (bool): Determines if the  given input file
                will be automatically overwritten.
            fn (str): The name of the input file
            path (str): The path to the input file
            verbose (bool): Determines if verbose execution is used
        """

        if inputs is not None and isinstance(inputs, list):
            self.inputs = inputs
        else:
            self.inputs = list()

        if isinstance(overwrite_flag, bool):
            self.overwrite_flag = overwrite_flag
        else:
            self.overwrite_flag = False

        if isinstance(verbosity, bool):
            self.verbose = verbosity

            if self.verbose:
                print(">>>> Initialized an InputFile object with attributes: ")
                print(self.__dict__)
        else:
            self.verbose = False

        self.fn, self.path = self.create(inpt_fn)

    def create(self, inpt_fn=None):
        """Creates or loads an input file into the InputFile object.

        If the user does not given an input file name, then the user is
        prompted to give a name. If the input file exists already, and the
        overwrite_flag is False, then the user is prompted to confirm
        overwriting of the input file. Once this is confirmed the entire
        file will be emptied. If the user elects to not overwrite the file,
        then they are prompted to input a new file name.

        Args:
            inpt_fn (str): The input file name.

        Attributes:
            fn (str): The input file name.
            path (str): The path to the input file.
        """
        if self.verbose:
            print("Creating an input file object and input file")

        new_inpt_prompt = "Please enter a new input file or exit [enter]? "

        if inpt_fn is None or inpt_fn is "":
            inpt_fn = input(new_inpt_prompt)

        # Checks if the file exists already and if so gives the user
        # the option to overwrite it; otherwise, it generates a new file
        if os.path.isfile(inpt_fn):
            if not self.overwrite_flag:
                print("Warning: The file {} already exists."
                      .format(color_file(inpt_fn)))

                overwrite = input("Would you like to overwrite {} [Y/N]: "
                                  .format(color_file(inpt_fn)))

                if overwrite is "":
                    sys.exit("Exiting")

                if re.match("(n|N)", overwrite):
                    print_directory(os.curdir)

                    inpt_fn = input(new_inpt_prompt)

                    if inpt_fn is "":
                        sys.exit("Exiting")

                    else:
                        return self.create(inpt_fn)

                elif re.match("(y|Y)", overwrite):
                    with open(inpt_fn, "w") as f:
                        f.truncate(0)
            else:
                if self.verbose:
                    print("Automatically overwriting {}..."
                          .format(inpt_fn))

                with open(inpt_fn, "w") as f:
                    f.truncate(0)

        if inpt_fn is "":
            sys.exit("Exiting")

        mkfileCmd = "touch {}".format(inpt_fn)
        sub.call(mkfileCmd.split())

        fn = inpt_fn
        path = os.path.join(os.curdir, inpt_fn)

        return fn, path

    def attributes(self):
        """Prints all of the attributes of the InputFile object"""
        if self.verbose:
            print("Printing attributes ...\n")

        print("Attributes of {} object {}:".format(self.__class__, self))

        print(self.__dict__)


class SummaryFile():
    """
    This object encapsulates the entire state of the summary file.

    A summary file is where we store the object attributes by
    serializing them.

    Summary files are stores in ./summaries.
    """

    def __init__(self, sum_fn=None, verbosity=False):
        """Initializes a SummaryFile object

        Args:
            sum_fn (str): The summary file name
            verbosity (bool): Turns on and off verbose ouput

        Attributes:
            verbose (bool): Determines if verbose execution is used
        """
        if isinstance(verbosity, bool):
            self.verbose = verbosity

            if self.verbose:
                print(">>> Initialized a SummaryFile object with attributes: ")
                print(self.__dict__)
        else:
            self.verbose = False

        self.fn, self.path, self.serial = self.create(sum_fn)

    def create(self, sum_fn=None):
        """Creates or loads a summary file

        If a summary file name is not provided then the user is prompted for one
        and the files inside of ./summaries is provided for reference. The summary
        file name must also end in either .json or .hdf5 in order to be created
        and compatible with serialization techniques implemented.

        Parameters:
            sum_fn (str): The summary file name.

        Attributes:
            fn (str): The input file name.
            path (str): The path to the input file.
            serial (str): The type of serialization for this summary file
        """
        if self.verbose:
            print("Creating an summary file object and summary file")

        # If sum_fn is not given, then we request the user
        sum_fn_prompt = "Please input a valid summary file or exit [enter]: "
        if sum_fn is None or sum_fn is "":
            print_directory("./summaries")

            sum_fn = input(sum_fn_prompt)

        if sum_fn is "":
            sys.exit("Exiting")

        serial = re.search("\\.(json|hdf5)", sum_fn)

        # If serial cannot be found or is not of the correct type
        # Then it request the user for a proper file name
        if serial is None:
            print("{}".format(sum_fn))

            print("Your file name must end in .json or .hdf5.")

            sum_fn = input("Enter a new file name or exit [enter]: ")

            if serial is "":
                sys.exit("Exiting")

            else:
                return self.create(sum_fn)

        path_to_file = os.path.join(os.curdir, "summaries", sum_fn)

        # If this summary file does not exist, create it.
        if not os.path.isfile(path_to_file):

            mkFileCmd = "touch {}".format(path_to_file)

            sub.call(mkFileCmd.split())

        path = os.path.dirname(path_to_file)
        fn = sum_fn
        serial = serial.group(1)

        return fn, path, serial

    def attributes(self):
        """Prints out all of the attributes of the SummaryFile object"""
        if self.verbose:
            print("Printing attributes ...\n")

        print("Attributes of {}".format(self))

        print(self.__dict__)



