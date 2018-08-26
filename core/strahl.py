########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: This will load the parameter files to be worked with.

Details: Calls the local private function _loadParamFile
    to load

To Do:

"""

__author__ = 'Joshua Swerdow'

import os
import sys
import subprocess
import numpy as np

from copy import deepcopy
from scipy.io import netcdf

from PyStrahl.utils import query
from PyStrahl.utils.auxillary import print_directory, parameter_lists
from PyStrahl.core.editor import (ParameterFileEditor,
                                  InputFileEditor,
                                  SummaryFileEditor)


# TODO:
# General:
#   Autocompletion to file entry?
#
# strahl_interface:
#   Add: ability to write in any strahl_interf input from cmd line or txt file
#       so every single input parameter for every function should be written
#       in the main file, but left untouched for the most part.
#   Make 'prompt' parts of code into some sort of query function that
#       can accept help (h, help, H, etc) inputs
#   Add HDF5 integration

# Parameters: Go through with Peter
#   Add all parameters to main, background files
#   Add/Create parameters for geometry/flux(?)
#   Fix parameters that have one header for multiple parameters: Parameter
#   Parameters with the same name within a file


def interface(main_fn=None, bckg_fn=None, geom_fn=None, flux_fn=None,
              inpt_fn=None, summ_fn=None, inputs=None,
              execute=False, verbosity=False):
    """
    Runs through an interactive, self-checking, interface for creating STRAHL
    input files.
    """
    if main_fn is None:
        file_name = ""

        if "-main_fn" in sys.argv:
            index = sys.argv.index("-main_fn")
            file_name = sys.argv[index + 1]

        main_fn = file_name

    if bckg_fn is None:
        file_name = ""

        if "-bckg_fn" in sys.argv:
            index = sys.argv.index("-bckg_fn")
            file_name = sys.argv[index + 1]

        bckg_fn = file_name

    if geom_fn is None:
        file_name = ""

        if "-geom_fn" in sys.argv:
            index = sys.argv.index("-geom_fn")
            file_name = sys.argv[index + 1]

        geom_fn = file_name

    if flux_fn is None:
        file_name = ""

        if "-flux_fn" in sys.argv:
            index = sys.argv.index("-flux_fn")
            file_name = sys.argv[index + 1]

        flux_fn = file_name

    if inpt_fn is None:
        file_name = ""

        if "-inpt_fn" in sys.argv:
            index = sys.argv.index("-inpt_fn")
            file_name = sys.argv[index + 1]

        inpt_fn = file_name

    if summ_fn is None:
        file_name = ""

        if "-summ_fn" in sys.argv:
            index = sys.argv.index("-summ_fn")
            file_name = sys.argv[index + 1]

        summ_fn = file_name

    if "-e" in sys.argv:
        execute = True

    if "-v" in sys.argv:
        verbose = True
    else:
        verbose = verbosity

    main_editor = None
    bckg_editor = None
    geom_editor = None
    flux_editor = None

    inpt_editor = None
    summ_editor = None

    parameter_editors = ParameterFileEditor.create_editors(main_fn, bckg_fn,
                                                           geom_fn, flux_fn,
                                                           verbosity=verbose)

    main_editor, bckg_editor, geom_editor, flux_editor = parameter_editors

    inpt_editor = InputFileEditor(inpt_fn=inpt_fn, verbosity=verbose)

    summ_editor = SummaryFileEditor(sum_fn=summ_fn, verbosity=verbose)

    for editor in parameter_editors:
        if editor is not None:
            editor.search_parameter_file()

    # Load the summary file into a new object and extract relevant features
    # into a set data structure
    summ_dict = None
    summ_on = list()
    summ_off = list()
    parameter_dict = dict()

    for editor in parameter_editors:
        if editor is not None:
            file = editor.parameter_file

            if verbose:
                print("Iterating through parameter files: {}"
                      .format(file.type))

            if summ_editor.loadable:
                summ_dict = summ_editor.get()

                sum_param = summ_dict[file.type]['param_dict']

                summ_on, summ_off = parameter_lists(sum_param)

            keys = ['fn', 'parameter_dict']
            param_keys = ['name', 'value', 'state']

            # Extract the parameters of a given parameter file
            file_dict = file.attribute_dictionary(keys, param_keys)

            file_param = file_dict['parameter_dict']

            # Extract relevant parameter file features into a
            # set data structure
            file_on, file_off = parameter_lists(file_param)

            # Perform operations to determine appropriate actions
            # Change the value: param_on & sum_on
            on_in_both = [param for param in file_on if param in summ_on]

            query_str = lambda x: ("Enter a new value for '{}' or ".format(x) +
                "keep the same value ({}) [enter]: "
                .format(sum_param[x]))

            if verbose:
                print("Iterating through parameters that are on in the " +
                      "summary and the parameter file.")

            checked = list()
            for param in on_in_both:
                if param not in checked:
                    group = file.in_group(param)

                    if group is not None:
                        file.groups.remove(group)

                        if verbose:
                            print("Iterating through parameters in group {}"
                                  .format(group))

                        vals = list()
                        for par in group:
                            # Evaluate the user input
                            val = query.evaluate(query_str(par))

                            # Change the value in the parameter file dictionary
                            file.parameter_dict[par].change_value(val)

                            # Append to our list
                            vals.append(val)

                            # Add to our checked list
                            checked.append(par)

                        # Change list to np.array and transpose for approriate
                        # writing format into STRAHL
                        val = np.array(vals).transpose()

                        # Add the val to the input files list of values
                        inpt_editor.add(val)

                    else:
                        val = query.evaluate(query_str(param))

                        if val is None:
                            val = sum_param[param]['value']

                        inpt_editor.add(val)

                        file.parameter_dict[param].change_value(val)

                        checked.append(param)

            # Change the value: param_on - sum_ff
            param_on_sum_off = [param for param in file_on
                                if param not in summ_off]

            if verbose:
                print("Iterating through parameters that are off in the " +
                      "summary and on in the parameter file.")

            query_str = lambda x: "Enter your new value for '{}': ".format(x)

            checked = list()
            for param in param_on_sum_off:
                if param not in checked:
                    group = file.in_group(param)

                    if group is not None:
                        file.groups.remove(group)

                        vals = list()
                        for par in group:
                            # Evaluate the user input
                            val = query.evaluate(query_str(par))

                            # Change the value in the parameter file dictionary
                            file.parameter_dict[par].change_value(val)

                            # Append to our input list
                            vals.append(val)

                            # Add to our checked list
                            checked.append(par)

                        # Change list to np.array and transpose for approriate
                        # writing format into STRAHL
                        val = np.array(vals).transpose()

                        # Add the val to the input files list of values
                        inpt_editor.add(val)

                    else:
                        val = query.evaluate(query_str(param))

                        inpt_editor.add(val)

                        file.parameter_dict[param].change_value(val)

                        checked.append(param)

            parameter_dict[file.type] = file.attribute_dictionary(keys,
                                                                  param_keys)

    # Clean-up
    if main_editor is None:
        print("Warning: There is no main parameter file editor.")
        main_fn = input("Please input a main file name: ")
    else:
        main_fn = main_editor.parameter_file.fn

    inpt_editor.write(main_fn)
    summ_editor.write(parameter_dict)

    # Execute strahl
    if execute:
        execute()


def quick_input_file(main_fn=None, inpt_fn=None, inputs=None,
                     verbose=False):
    """
    WARNING WARNING WARNING
    PLEASE READ BEFORE USE
    Quick way to (re) write an input file given a list of inputs
    in the order that you know they should be read in! If the given
    input file already exists this method will remove the contents
    and rewrite it given the new inputs. Please be careful as
    there are NO SAFETY protocols in place to ensure the input file you
    give are indeed input files (i.e. one could delete the wrong type
    of file if passed through this method). You have been warned.
    """

    if main_fn is None or inpt_fn is None or inputs is None:
        sys.exit()

    with open(inpt_fn, "w+") as file:
        file.truncate()

    inpt_editor = InputFileEditor(inpt_fn=inpt_fn, inputs=inputs,
                                  overwrite_flag=True,
                                  verbosity=verbose)

    inpt_editor.write(main_fn)


def run(inpt_fns=None, strahl_cmd=None, verbose=False):
    """
    Runs strahl using an input file to fill in the parameters that have been
    commented out from parameter files.

    Parameters:
        * **inpt_fns** [list/str|None]: name(s) of the input file in a list or
            string
        * **strahl_cmd** [str|"./strahl"]: the command to run strahl which can
            be replaced with different tags that are described in the user
            manual.

    """
    if strahl_cmd is None:
        strahl_cmd = "./strahl"

    if not strahl_cmd.startswith("./strahl"):
        sys.exit("Error: strahl command is of wrong format " +
                 "it should start with ./strahl")

    if inpt_fns is None:
        print_directory(os.curdir)

        prompt = "Enter and input file(s) seperated by space or exit [enter]: "
        inpt_fns = input(prompt)

        if inpt_fns is "":
            sys.exit("Exiting")

    # Converts single and multi input strings to lists
    if isinstance(inpt_fns, str):
        inpt_fns = inpt_fns.split()

    elif not isinstance(inpt_fns, list):
        sys.exit("TypeError: inpt_fns must be list or list of strings.")

    for inpt_fn in inpt_fns:
        if not os.path.isfile("./" + inpt_fn):
            print("Error: ./{} is not a valid file".format(inpt_fn))

            inpt_fn = input("Enter the correct file or exit [enter]: ")

            if inpt_fn is "":
                sys.exit("Exiting")

        with open(inpt_fn, 'r') as f:
            if verbose:
                print("Executing: {} < {}".format(strahl_cmd, inpt_fn))

            process = subprocess.Popen(strahl_cmd.split(), stdin=f,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True)

            output, errmsg = process.communicate()

            if verbose:
                print("Output:\n{}".format(output))

            if errmsg is not "":
                print("Error:\n{}".format(errmsg))


def extract_results(result="Festrahl_result.dat",
                    variables=None, dimensions=None, attributes=None,
                    verbose=False):
    """
    Extracts results from appropriate netcdf groups given a list of item names
    for any of the groups ('variables', 'dimensions', '_attributes')

    Parameters:
        * **result** [str]: A file path to the netCDF file
        * **variables** [list|None]: A list of items names that must be
            extracted from the variables group.
        * **attributes** [list|None]: A list of items names that must be
            extracted from the attributes group.
        * **dimensions** [list|None]: A list of items names that must be
            extracted from the dimensions group.
    Returns:
        A dictionary containing the group name and a dictionary of
        the items requested from the group object for each group given
        to this function.
    """
    path = "result"
    if os.path.dirname(result) != path:
        result = os.path.join(path, os.path.basename(result))

    data = dict()

    if variables is not None:
        if isinstance(variables, (list, np.ndarray)):
            group_items = _extract_items_from_group(result,
                                                    "variables",
                                                    variables,
                                                    verbose=verbose)

            data['variables'] = group_items

        else:
            print("Error: variables must be a list.\n")

    if dimensions is not None:
        if isinstance(dimensions, (list, np.ndarray)):
            group_items = _extract_items_from_group(result,
                                                    "dimensions",
                                                    dimensions,
                                                    verbose=verbose)

            data['dimensions'] = group_items

        else:
            print("Error: dimensions must be a list.\n")

    if attributes is not None:
        if isinstance(attributes, (list, np.ndarray)):
            group_items = _extract_items_from_group(result,
                                                    "attributes",
                                                    attributes,
                                                    verbose=verbose)

            data['attributes'] = group_items

        else:
            print("Error: attributes must be a list.\n")

    return data


def _extract_items_from_group(result, group_type, item_names, verbose=False):
    """
    Extracts items from a netcdf group stored within STRAHL result
    files.

    Parameters:
        * **result** [str]: A file path to the netCDF file

        * **verbose** [bool|False]: Determines if verbose execution is used
        * **all** [bool|False]: Determines if entire information of netCDF file
            should be printed

    Returns:
        A dictionary of netCDF item name and netCDF items for every
        list of items. If a variable cannot be found in the result file,
        then None is returned in its place.
    """

    group_data = None

    if os.path.isfile(result):
        with netcdf.netcdf_file(result, 'r', mmap=True) as file:
                if group_type is "variables":
                    group_ = deepcopy(file.variables)

                if group_type is "dimensions":
                    group_ = deepcopy(file.dimensions)

                if group_type is "attributes":
                    group_ = deepcopy(file._attributes)

                group_.name = group_type

                group_data = _extract_items(group_, item_names,
                                            verbose=verbose)

    else:
        sys.exit("File does not exist. Exiting.\n")

    return group_data


def _extract_items(group_, item_names, verbose=False):
    """
    Extracts specified items from a netcdf group object.

    Parameters:
        * **group_** [netcdf obj]: A netcdf object containing items
        * **item_names** [list|None]: A list of item names to extract
            from the group_ object
        * **verbose** [bool|False]: determines if verbose execution is used

    Returns:
        A dictionary of the items name and items equivalent
        to the length of the item_names. If a variable cannot be found
        in the result file, then None is returned in its place.
    """
    data = dict()

    if len(item_names) == 0:
        print("Warning: item_names for '{}' group is empty. "
              .format(group_.name) + "Nothing returned.\n")

    for item_name in item_names:
        if item_name in group_.keys():
            item_ = group_.get(item_name)

            if isinstance(item_, netcdf.netcdf_variable):
                item_.name = item_name

                if verbose:
                    print("Updating dictionary with {}.".format(item_name))
                    print("{} has attributes:".format(item_))
                    print("\t--> Name: {}".format(item_name))
                    print("\t--> Units: {}".format(item_.units))
                    print("\t--> Shape: {}".format(item_.shape))
                    print("\t--> Dimensions: {}".format(item_.dimensions))
                    print("\t--> Data: {}".format(item_.data))
                    print("\n")

            elif isinstance(item_, int):
                if verbose:
                    print("Updating dictionary with {}.".format(item_name))
                    print("{} has attributes:".format(item_name))
                    print("\t--> Name: {}".format(item_name))
                    print("\t--> Data: {}".format(item_))
                    print("\n")

            if item_name in data.keys():
                print("""{var} was extracted already or is the name for
                    multiple variables. {} will be overwritten."""
                    .format(var=item_name))

            data[item_name] = item_

        else:
            print("Warning: '{}' is not in the data set.\n"
                  .format(item_name))

    return data


def extract_all(result="Festrahl_result.dat"):
    """
    Extracts and prints all the item names and shapes from the
    variables, dimensions, and _attributes groups within the
    given result file.
    """

    path = "result"
    if os.path.dirname(result) != path:
        result = os.path.join(path, os.path.basename(result))

    groups_ = list()
    if os.path.isfile(result):
        with netcdf.netcdf_file(result, 'r', mmap=True) as file:
                    variables_ = deepcopy(file.variables)
                    variables_.name = "variables"
                    groups_.append(variables_)

                    dimensions_ = deepcopy(file.dimensions)
                    dimensions_.name = "dimensions"
                    groups_.append(dimensions_)

                    attributes_ = deepcopy(file._attributes)
                    attributes_.name = "attributes"
                    groups_.append(attributes_)
    else:
        sys.exit("File does not exist. Exiting.\n")

    for group_ in groups_:
        print("'{}' items and shapes:".format(group_.name))
        for name, item in group_.items():

            if hasattr(item, 'shape'):
                shape = item.shape
            else:
                shape = item

            print("\t--> {}..............{}".format(name, shape))
        print("\n")


def analyze(diff=None, conv=None, data_file=None, variables=None,
            noises=None, lstsq=None, mcmc=None, gp=None,
            verbose=None):

    if verbose is None:
        if "-v" in sys.argv:
            index = sys.argv.index("-v")
            verbose = True
        else:
            verbose = False

    if diff is None:
        diffusion_profile = None

        # Must be entered as comma seperated values with no spaces
        if "-diff" in sys.argv:
            index = sys.argv.index("-diff")
            diffusion_profile = sys.argv[index + 1]
    else:
        diffusion_profile = diff

    if diffusion_profile is not None:
        if not isinstance(diffusion_profile, list):
            diffusion_profile = [diffusion_profile]

        profiles.append(diffusion_profile)

    if conv is None:
        convention_profile = None

        # Must be entered as comma seperated values with no spaces
        if "-conv" in sys.argv:
            index = sys.argv.index("-conv")
            convention_profile = sys.argv[index + 1]
    else:
        convention_profile = conv

    if convention_profile is not None:
        if not isinstance(convention_profile, list):
            convention_profile = [convention_profile]

        profiles.append(convention_profile)

    if verbose:
        print("Interfacing with strahl...")

    interface(main_fn="example", inpt_fn="test", summ_fn="test.json",
              verbosity=False)

    if data_file is None:
        data_file = "Festrahl_result.dat"

        if "-data_file" in sys.argv:
            index = sys.argv.index("-data_file")
            data_file = sys.argv[index + 1]

    if variables is None:
        variables = ["time", "diag_rad_line"]

        if "-variables" in sys.argv:
            index = sys.argv.index("-variables")
            variables = [var.strip() for var in sys.argv[index + 1].split(",")]

    if verbose:
        print("Extracting variables {} from data file {}".format(variables, data_file))

    # SPLIT INTO INDEP AND DEP VARIABLE SETS
    # HELPS WITH NOISE
    # PERHAPS HELPS WITH FITTING SINCE WE WILL KNOW HOW MANY INDEP/DEP THERE ARE AUTOMATICALLY
    data = get_netCDF_variables(data_file, variables, verbose)

    fitting_algorithms = list()
    if lstsq is None:
        if "-lstsq" in sys.argv:
            lstsq = "Least Square"
            fitting_algorithms.append(lstsq)

    if mcmc is None:
        if "-mcmc" in sys.argv:
            mcmc = "Markov Chain Monte Carlo"
            fitting_algorithms.append(mcmc)

    if gp is None:
        if "-gp" in sys.argv:
            gp = "Gaussian Process"
            fitting_algorithms.append(gp)

    if noises is None:
        noises = []

        if "-noises" in sys.argv:
            index = sys.argv.index("-noises")
            noises = [noise.strip() for noise in sys.argv[index + 1].split(",")]

    if verbose:
        print("Fitting with the following algorithms: {}".format(fitting_algorithms))
        print("Using the following noises: {}".format(noises))

    data = make_noisy(data, noises, verbose)

    fit(data, fitting_algorithms)
