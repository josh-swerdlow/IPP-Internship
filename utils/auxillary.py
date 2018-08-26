import os
import re
import numpy
import termcolor

from PyStrahl.core.parameters import Parameter


def print_directory(path):
    files = os.listdir(path)

    print("Map: {}, {}, {}, {}\n\n". format(color_file("files"),
        color_dir("directories"), color_py("python code"),
        color_pckl("summary files")))

    for file in files:
        f = os.path.join(path, file)

        if os.path.isfile(f):
            # Python (.py) files
            if re.match(".*.py", file):
                print("  {}  ".format(color_py(file)))
            elif re.match(".*.pckl", file):
                print("  {}  ".format(color_pckl(file)))
            else:
                print("  {}  ".format(color_file(file)))

        elif os.path.isdir(f) is True:
            print("  {}  ".format(color_dir("./" + file)))

    print("\n")


def color_file(file):
    return color(file, 'blue')


def color_dir(dir):
    return color(dir, 'red')


def color_py(pyFile):
    return color(pyFile, 'cyan')


def color_pckl(pickle):
    return color(pickle, 'green')


def color(file, color):
    """
    Colors: red, green, blue, yellow, magenta, cyan, white
    """
    return termcolor.colored(file, color)


def generate_dictionary(dic):
    """
    args should be a list of string arguments to identify what attributes
    one would like to keep in the dictionary. this list should have the
    attributes for every object expected to be found
    """
    D = dict()

    if not isinstance(dic, dict):
        dic = dic.__dict__

    for key, val in dic.items():
        if isinstance(val, dict):
            val = generate_dictionary(val)

        if isinstance(val, Parameter):
            val = generate_dictionary(val.__dict__)

        if isinstance(val, numpy.ndarray):
            val = val.tolist()

        D[key] = val

    return D


def parameter_lists(dic=None):
    """
    Returns the entire parameter set for the on and off state
    parameter sets seperately
    """
    params_on = list()
    params_off = list()

    if isinstance(dic, dict) and len(dic) != 0:
        params, vals = zip(*dic.items())

        params_on = [key for (key, val) in zip(params, vals) if val['state']]
        params_off = [param for param in params if param not in params_on]

    return params_on, params_off
