########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 08/06/2018             #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: This will perform sensitivty analysis on the D and V profiles.

Details:

To Do:

"""

__author__ = 'Joshua Swerdow'

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


from scipy.io import netcdf
from copy import deepcopy
from strahl_interface import interface


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


def get_netCDF_variables(result, variables=None, verbose=False, all=False):
    """
    Gets a list of netCDF variables from the result file with the ability to display
    information about the entire netCDF result file.

    Parameters:
        * **result** [str]: A file path to the netCDF file
        * **variables** [list|None]: A list of variable names to extract from the
            result file
        * **verbose** [bool|False]: determines if verbose execution is used
        * **all** [bool|False]: determines if entire information of netCDF file
            should be printed

    Returns:
        A list of netCDF variables equivalent to the length of the variables
        parameter. If a variable cannot be found in the result file, then None
        is returned in its place.
    """
    path = "result"
    if os.path.dirname(result) != path:
        result = os.path.join(path, os.path.basename(result))

    signal = None
    dimension = None
    attribute = None

    if os.path.isfile(result):
        with netcdf.netcdf_file(result, 'r', mmap=True) as file:

            signal = deepcopy(file.variables)
            dimension = deepcopy(file.dimensions)
            attribute = deepcopy(file._attributes)
    else:
        sys.exit("File does not exist. Exiting.")

    if variables is None:
        print("Warning: No variables are being extracted from {}".format(result))
        variables = list()

    data = None

    if all:
        print('\nDimension names and shapes:')
        for key, value in dimension.items():
            print("{}...........................{}".format(key, value))

        print('\nAttribute names and shapes:')
        for key, value in attribute.items():
            print("{}...........................{}".format(key, value))

        print('\nVariable names and shapes:')
        for key, value in signal.items():
            print("{}...........................{}".format(key, value.shape))

    data = list()
    for variable in variables:
        if variable in signal.keys():
            variable_copy = signal.get(variable)
            variable_copy.name = variable

            if verbose:
                print("{} has attributes:".format(variable))
                print("\t Units: {}".format(variable_copy.units))
                print("\t Shape: {}".format(variable_copy.shape))
                print("\t Dimensions: {}".format(variable_copy.dimensions))
                print("\t Data: {}".format(variable_copy.data))
                print("\t Name: {}".format(variable_copy.name))

                print("Append {} to data list.".format(variable))

            data.append(variable_copy)

        else:
            print("WARNING: '{}' is not in the data set.".format(variable))

    return data


def fit(data, fitting_algorithms):
    pass


def make_noisy(data, noises=None, verbose=False):
    """
    Takes the given data and gives them artificial noise

    Parameters:
        * **data** [list(netCDF variable)]: A list of netCDF variables
        * **noises** [list/str|None]: A list of keywords for each data variable
            to determine how to handle the noise for each data variable
        * **verbose** [bool|False]: determines if verbose execute is used

    Returns:
        A list of new netCDF variables for noisy data
    """
    if noises is None:
        noises = list()

    if isinstance(noises, str):
        if verbose:
            print("Converting noises from str to list.")

        noises = [noises]


    numb_noises = len(noises)
    numb_data = len(data)

    if numb_noises != numb_data:
        if verbose:
            print("There are {} noise keyword(s) and {} data variable(s)."
                  .format(numb_noises, numb_data))
            print("\t Noise keywords: {}".format(noises))

        if numb_noises == 1:
            if verbose:
                print("Using the noise keyword '{}' for all data variables"
                      .format(noises[0]))

            noises = [noises[0] for _ in data]

        else:
            sys.exit("Error: Mismatched list lengths.")

    for dat, noise in zip(data, noises):
        if verbose:
            print("Making '{}' noisy with {} distribution.".format(dat.name, noise))

        ## FINISH ADDING NOISE TO DATA
        if noise is "poisson":
            pass
        else:




def plot_d_and_v(data_file, times=None):
    """
    Plots the D and v profiles for a given data_file at the given time indices.

    Parameters:
        * **data_file** [str]: A file path to the netCDF file
        * **times** [list/int|None]: A list of time indices to plot profile at
    """

    D = "anomal_diffusion"
    v = "anomal_drift"
    rho = "rho_poloidal_grid"
    t = "time"

    D, v, rho, t = get_netCDF_variables(data_file, [D, v, rho, t], verbose=True)

    if times is None or not isinstance(times, (list, np.ndarray)):
        times = [0]

    if isinstance(times, int):
        times = [times]

    for time in times:
        # Generate for plots
        t_label = t[time]

        # Convert units
        # cm^2/s --> m^2/s
        diffusion = D[time, :] * (10**-4)
        # cm/s --> m/s
        drift = v[time, :] * (10**-2)

        plt.figure("Diffusion Profile")
        plt.scatter(rho.data, diffusion)
        plt.xlabel("rho (r/a)")
        plt.ylabel("D [m^2/s]")
        plt.title("Diffusion profile at t={} s".format(t_label))

        plt.figure("Drift Profile")
        plt.scatter(rho.data, drift)
        plt.xlabel("rho (r/a)")
        plt.ylabel("v [m/s]")
        plt.title("Drift profile at t={} s".format(t_label))

        plt.show()


if __name__ == '__main__':
    # analyze(diff="1, 2, 3", conv="9, 9, 9")
    analyze(verbose=True, variables=["time", "diag_lines_radiation"], noises="gaussian")
    # data = get_netCDF_variables("result/Festrahl_result.dat", all=True)
    # make_noisy(data, True)
    # plot_d_and_v("result/Festrahl_result.dat", times=[1, 2, 3])
    # make_noisy(["time"], ["poisson"], True)
