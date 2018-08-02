from source.parameters import Parameter

import termcolor
import numpy
import os
import re


def print_dirContents(path):
    files = os.listdir(path)

    print("Map: {}, {}, {}, {}\n\n". format(colorFile("files"),
        colorDir("directories"), colorPy("python code"),
        colorPckl("summary files")))

    for file in files:
        f = os.path.join(path, file)

        if os.path.isfile(f):
            # Python (.py) files
            if re.match(".*.py", file):
                print("  {}  ".format(colorPy(file)))
            elif re.match(".*.pckl", file):
                print("  {}  ".format(colorPckl(file)))
            else:
                print("  {}  ".format(colorFile(file)))

        elif os.path.isdir(f) is True:
            print("  {}  ".format(colorDir("./" + file)))

    print("\n")


def colorFile(file):
    return color(file, 'blue')


def colorDir(dir):
    return color(dir, 'red')


def colorPy(pyFile):
    return color(pyFile, 'cyan')


def colorPckl(pickle):
    return color(pickle, 'green')


def color(file, color):
    """
    Colors: red, green, blue, yellow, magenta, cyan, white
    """
    return termcolor.colored(file, color)


# def binaryInputDecision(prompt, exit_param, fun):
#     answer = input(prompt)

#     if answer is exit_param:
#         sys.exit("Exiting")
#     else:
#         fun(answer)

def generateDictionary(dic):
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
            val = generateDictionary(val)

        if isinstance(val, Parameter):
            val = generateDictionary(val.__dict__)

        if isinstance(val, numpy.ndarray):
            val = val.tolist()

        D[key] = val

    return D


def parameter_sets(dic=None):
    """
    Returns the entire parameter set and the on and off state
    parameter sets seperately.
    """

    keys, vals = zip(*dic.items())

    params = set(keys)

    params_on = {key for (key, val) in zip(keys, vals) if val['state']}
    params_off = params - params_on

    return params_on, params_off


