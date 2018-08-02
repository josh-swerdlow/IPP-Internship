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

import sys

import strahl_run
import numpy as np
import source.interface.query as query
import source.interface.auxillary as aux

from source.interface.editor import ParameterFileEditor, InputFileEditor, SummaryFileEditor


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
#   Fix parameters that have one header for multiple parameters: Parameters obj?
#   Parameters with the same name within a file


def interface(main_fn=None, bckg_fn=None, geom_fn=None, flux_fn=None,
              inpt_fn=None, summ_fn=None, execute=False, verbosity=False):
    """
    Runs through an interactive, self-checking, interface for creating STRAHL
    input files.
    """
    if main_fn is None:
        main_fn = "op12a_171122022_FeLBO3_test"

    if bckg_fn is None:
        bckg_fn = ""

    if geom_fn is None:
        geom_fn = ""

    if flux_fn is None:
        flux_fn = ""

    if inpt_fn is None:
        inpt_fn = "test"

    if summ_fn is None:
        summ_fn = "summ_test.json"

    if "-e" in sys.argv:
        execute = True

    if "-v" in sys.argv:
        verbose = True
    else:
        verbose = verbosity

    main_editor = None
    bckg_editor = None
    inpt_editor = None
    summ_editor = None
    # FIND A WAY TO DEAL WITH ENTERING ONLY SOME OF THE PARAMETER FILE
    # NAMES AND HOW TO UNPACK THEM CORRECTLY SO ONE DOESN'T NEED TO
    # CHANGE THE CODE ALL THE TIME (i.e. if one only changes the main file
    # then this should be able to run, even if they don't want to look
    # at the background file)
    parameter_editors = ParameterFileEditor.create_editors(main_fn,
                                                           bckg_fn,
                                                           verbosity=verbose)
    main_editor = parameter_editors

    inpt_editor = InputFileEditor(inpt_fn=inpt_fn, verbosity=verbose)

    summ_editor = SummaryFileEditor(sum_fn=summ_fn, verbosity=verbose)

    for editor in parameter_editors:
        editor.search_parameter_file()

    # Load the summary file into a new object and extract relevant features
    # into a set data structure
    summ_dict = None
    summ_on = set()
    summ_off = set()
    parameter_dict = dict()

    for editor in parameter_editors:
        file = editor.parameter_file

        if verbose:
            print("Iterating through parameter files: {}".format(file.type))

        if summ_editor.loadable:
            summ_dict = summ_editor.get()

            sum_param = summ_dict[file.type]['param_dict']

            summ_on, summ_off = aux.parameter_sets(sum_param)

        keys = ['fn', 'parameter_dict']
        param_keys = ['name', 'value', 'state']

        # Extract the parameters of a given parameter file
        file_dict = file.attribute_dictionary(keys, param_keys)

        param = file_dict['parameter_dict']

        # Extract relevant parameter file features into a set data structure
        file_on, file_off = aux.parameter_sets(param)

        # Perform set operations to determine appropriate actions

        # Change the value: param_on & sum_on
        on_in_both = file_on & summ_on

        query_str = lambda x: ("Enter a new value for '{}' or ".format(x) +
            "keep the same value ({}) [enter]: "
            .format(sum_param[x]))

        # TODO: PUT A CHECK FOR THE NUMBER OF VALUES EXPECTED
        # TODO: MAKE EVERYTHING INTO A GROUPED OBJECT AND REWRITE CODE
        #   TO GENERALIZE FOR ANY DIMENSIONAL OBJECT!
        if verbose:
            print("Iterating through parameters that are on in the summary " +
                "and the parameter file.")

        groups = list()
        for param in on_in_both:
            group = file.in_group(param)

            if group is not None:
                file.groups.remove(group)

                if verbose:
                    print("Iterating through parameters in group {}".format(group))

                vals = list()
                for par in group:
                    # Evaluate the user input
                    val = query.evaluate(query_str(par))

                    # Change the value in the parameter files dictionary
                    file.parameter_dict[par].change_value(val)

                    # Append to our list
                    vals.append(val)

                    # Remove from set to not iterate over again
                    on_in_both.remove(par)

                # Change list to np.array and transpose for approriate
                # writing format into STRAHL
                val = np.array(vals).transpose()

                # Add the val to the input files list of values
                inpt_editor.add(val)

                groups.append(groupobject)

        for group in groups:
            for param in group:
                param_on_sum_off.remove(param)

        for param in on_in_both:
            val = query.evaluate(query_str(param))

            if val is None:
                val = sum_param[param]['value']

            inpt_editor.add(val)

            file.parameter_dict[param].change_value(val)

        # Change the value: param_on - sum_ff
        param_on_sum_off = file_on - summ_off

        if verbose:
            print("Iterating through parameters that are off in the summary " +
                "and on in the parameter file.")

        query_str = lambda x: "Enter your new value for '{}': ".format(x)

        groups = list()
        for param in param_on_sum_off:
            group = file.in_group(param)

            if group is not None:
                file.groups.remove(group)

                vals = list()
                for par in group:
                    # Evaluate the user input
                    val = query.evaluate(query_str(par))

                    # Change the value in the parameter files dictionary
                    file.parameter_dict[par].change_value(val)

                    # Append to our list
                    vals.append(val)

                # Change list to np.array and transpose for approriate
                # writing format into STRAHL
                val = np.array(vals).transpose()

                # Add the val to the input files list of values
                inpt_editor.add(val)

                groups.append(group)

        for group in groups:
            for param in group:
                param_on_sum_off.remove(param)

        for param in param_on_sum_off:
            val = query.evaluate(query_str(param))

            inpt_editor.add(val)

            file.parameter_dict[param].change_value(val)

        parameter_dict[file.type] = file.attribute_dictionary(keys, param_keys)

    # Clean-up
    inpt_editor.write(main_fn)
    summ_editor.write(parameter_dict)

    # Execute strahl
    if execute:
        strahl_run()


if __name__ == '__main__':
    interface()
