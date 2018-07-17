# This will incorporate interacting with the user, fixing the files,
# and finally executing the strahl file
from loadParamFiles import loadParamFiles
from searchParameterFile import searchParameterFile

from editor import inputFileEditor
from editor import summaryFileEditor

from strahl_run import strahl_run

import auxillary as aux
import numpy as np

# TODO:
# strahl_interface:
#   Add: verbose flag
#   Add: run immediately after generating flag
#   Add: ability to write in any strahl_interf input from cmd line or txt file
#       so every single input parameter for every function should be written
#       in the main file, but left untouched for the most part.
#   Make 'prompt' parts of code into some sort of query function that
#       can accept help (h, help, H, etc) inputs
#   Clean up:
#       Write everything to file
#   Nested lists
#   Numpy arrays

# loadParamFiles:
#   Add: ability to enter file names if file is not given

# Parameters:
#   Add: dtype, vals, help attributes
#       dtype: acceptable data types in the form of a list (what about np dtypes?)
#       vals: if there are possible vals ('interp') then list them here
#       help: a string that is generated to describe the parameter and the vals
#   Add all parameters to main, background files
#   Add/Create parameters for geometry/flux(?)
#   Fix parameters that have one header for multiple parameters

#


def main():

    mainFileName = "op12a_171122022_FeLBO3"
    backgroundFileName = "pp22022.2"
    geomFileName = ""
    fluxFileName = ""
    parameter_files = loadParamFiles(mainFileName,
        backgroundFileName, geomFileName, fluxFileName)

    # Load parameter states
    searchParameterFiles(parameter_files)

    # Load input and summary editors
    inpt = None
    summ = None
    inpt_editor = inputFileEditor(inpt_fn=inpt)
    summ_editor = summaryFileEditor(sum_fn=summ)

    keys = ['fn', 'paramDict']
    param_keys = ['name', 'value', 'state']
    parameter_dict = {}

    for file in parameter_files:
        # Extract the parameters of a given parameter file
        file_dict = file.attribute_dictionary(keys, param_keys)

        param = file_dict['paramDict']

        # Extract relevant parameter file features into a set data structure
        file_on, file_off = aux.parameter_sets(param)

        # Load the summary file into a new object and extract relevant features
        # into a set data structure
        summ_dict = None
        summ_on = set()
        summ_off = set()

        if summ_editor.loadable:
            summ_dict = summ_editor.get()

            sum_param = summ_dict[file.type]['paramDict']

            summ_on, summ_off = aux.parameter_sets(sum_param)

        # Perform set operations to determine appropriate actions
        # Ask to change the value: param_on & sum_on
        on_in_both = file_on & summ_on

        for param in on_in_both:
            print("The value of {} was previously: {}".
                  format(param, sum_param[param]['value']))

            print("The follow request will be evaluated like python code!\n" +
                "As such, don't forget to put \"\" around strings.\n" +
                "In addition, the numpy (np) module can be used.")

            val = input("Would you like to change the value [<value>|enter]? ")

            if val is "":
                val = sum_param[param]['value']

            else:
                val = eval(val,
                          {'__builtins__': None, "np": np},
                          {'__builtins__': None}
                           )

            inpt_editor.add(val)

            file.paramDict[param].newValue(val)

        # Change the value: param_on - sum_ff
        param_on_sum_off = file_on - summ_off

        for param in param_on_sum_off:
            print("The follow request will be evaluated like python code!\n" +
                "As such, don't forget to put \"\" around strings.\n" +
                "In addition, the numpy (np) module can be used.")

            val = input("Enter your new value for {}: ".format(param))
            val = eval(val,
                      {'__builtins__': None, "np": np},
                      {'__builtins__': None}
                       )

            inpt_editor.add(val)

            file.paramDict[param].newValue(val)

        parameter_dict[file.type] = file.attribute_dictionary(keys, param_keys)


        print(inpt_editor.__dict__)
        file.attributes(True)

    # Clean-up
    inpt_editor.write()
    summ_editor.write(parameter_dict)

    # Execute strahl
    strahl_run()


if __name__ == '__main__':
    main()
