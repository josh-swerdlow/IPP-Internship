########################################
# File name: math.py                   #
# Author: Joshua Swerdow               #
# Date created: 8/21/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

__doc__ = """This package provides essential mathematical methods
             required to execute the strahl analysis"""

__author__ = "Joshua Swerdow"


import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import netcdf_variable


def charge_state_netcdf(variable_, charge_index):
    """Extracts a charge state slice of a netcdf variable's data

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        charge_index (int): The index of the "diagnostik_lines"
            dimension to use

    Returns:
        A simple array slice of the variable_.data for the given index
        of the "diagnostik_lines" dimension.
    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    dims = variable_.dimensions

    charge_str = "diagnostik_lines"
    charge_axis = dims.index(charge_str)

    data = variable_.data

    len_charge_axis = data.shape[charge_axis]
    if charge_index <= len_charge_axis - 1:
        charge_data = data.take(charge_index, axis=charge_axis)

    else:
        print("Index out of bounds error.")
        sys.exit("Cannot index into {} for array of length {}."
                 .format(charge_index, len_charge_axis))

    # Rewrite data and dimension
    variable_.data = charge_data
    variable_.dimensions = tuple([dim for dim in dims if dim != charge_str])

    return charge_data


def integrate_netcdf(variable_, integrat_dim):
    """Performs naive integration of a dimension on a netcdf variable's data.

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        integrat_dim (str): The dimension in the variable
            that is to be integrated over.

    Returns:
        A simple sum across the dimension or exits of cannot find
        the dimension.

    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    data = variable_.data
    dims = variable_.dimensions
    units = variable_.units.decode()

    if integrat_dim in dims:
        integrat_axis = dims.index(integrat_dim)

        integrated_data = np.sum(data, axis=integrat_axis)

        integrated_units = units.replace("/cm**3", "")
    else:
        sys.exit("Exiting: Cannot find '{}' in '{}' dimensions."
              .format(integrat_dim, variable_.name))

    # Rewrite data and dimensions of variable_
    variable_.data = integrated_data
    variable_.dimensions = tuple([dim for dim in dims if dim != integrat_dim])
    variable_.units = integrated_units.encode(encoding='utf-8')

    return integrated_data


def scale_netcdf(variable_, signal_scale, background_scale):
    """Scales a netcdf variable's data.

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        signal_scale (int): The value that is used to magnify the signal

    Returns:
        The scaled array of variables_.data
    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    data = variable_.data
    units = variable_.units.decode()

    # Scale the profile by some factor
    if signal_scale > 0:
        scaled_data = np.multiply(signal_scale, data)
        scaled_units = "({}) * {}".format(units, signal_scale)
    else:
        sys.exit("signal_scale must be > 0: {}".format(signal_scale))

    if background_scale > 0:
        scaled_data += background_scale
    else:
        sys.exit("background_scale must be > 0: {}".format(background_scale))

    # Rewrite data and units
    variable_.data = scaled_data
    variable_.units = scaled_units.encode(encoding='utf-8')

    return scaled_data


def noise_netcdf(variable_, noise_scale):
    """Adds scaled Gaussian noise to a netcdf variables data

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        noise_scale (int): The value that is use the magnify the noise

    Returns:
        A tuple of the noisy scaled signal and the noise added to the
        signal.
    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    data = variable_.data

    # Calculate sigmas
    sigmas = np.sqrt(data)

    # Calculate noise with a normal distribution centered at 0.0
    noise = noise_scale * np.random.normal(scale=sigmas, size=data.shape)
    noisey_data = data + noise

    # Rewrite data and dimension
    variable_.data = noisey_data

    return noisey_data, sigmas


def generate_signal(variable_, charge_index, integrat_dim,
                    signal_scale, background_scale, noise_scale):
    """Selects a charge state, naively integrates over a dimension,
    scales, and then finally adds scaled poisson noise to
    a netcdf variables data

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        integrat_dim (str): The dimension in the variable
            that is to be integrated over
        charge_index (int): The index of the "diagnostik_lines"
            dimension to use
        signal_scale (int): The value that is used to magnify the signal
        noise_scale (int): The value that is use the magnify the noise

    Returns:
        A tuple of the profile (integrated data), scaled profile,
        noisey experimental signal (scaled profile + noise) and the
        sigmas of the scaled profile.
    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    charge_state_netcdf(variable_, charge_index=charge_index)

    profile = integrate_netcdf(variable_, integrat_dim=integrat_dim)

    scaled_profile = scale_netcdf(variable_, signal_scale=signal_scale,
                                  background_scale=background_scale)

    signal, sigma = noise_netcdf(variable_, noise_scale=noise_scale)

    return (profile, scaled_profile, signal, sigma)


def generate_profile(variable_, charge_index, integrat_dim, signal_scale,
                     background_scale):
    """Selects a charge state, naively integrates over a dimension,
    and scales a netcdf variables data.

    Be warned, this method overwrites the appropriate attributes of the
    netcdf variable.

    Args:
        variable_ (:obj: netcdf_variable): A netcdf variable object
        integrat_dim (str): The dimension in the variable
            that is to be integrated over
        charge_index (int): The index of the "diagnostik_lines"
            dimension to use
        signal_scale (int): The value that is used to magnify the signal

    Returns:
        A  tuple of the profile and scaled profile for the integrated
        netcdf variable.
    """
    if not isinstance(variable_, netcdf_variable):
        sys.exit("variable_ is not an instance netcdf_variable.")

    charge_state_netcdf(variable_, charge_index=charge_index)

    profile = integrate_netcdf(variable_, integrat_dim=integrat_dim)

    scaled_profile = scale_netcdf(variable_, signal_scale=signal_scale,
                                  background_scale=background_scale)

    return (profile, scaled_profile)


