import os
import termcolor
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
