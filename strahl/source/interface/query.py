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

import numpy

# This will serve as a a module of queries for the user


def evaluate(query="Please input a value: "):

    print("\nThe following request will be evaluated like python code!\n" +
          "As such, don't forget to put \"\" around strings " +
          "and , between values.\n" +
          "In addition, the numpy (np) module can be used.")

    value = eval(input(query),
                {'__builtins__': None, "np": numpy},
                {'__builtins__': None}
                 )

    if value is "":
        value = None

    if isinstance(value, tuple):
        value = list(value)

    return value


