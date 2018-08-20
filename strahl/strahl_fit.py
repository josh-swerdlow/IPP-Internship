import sys

import strahl
import numpy as np
import matplotlib.pyplot as plt

from data import Splines, Residual
from models import Least_Square, Markov_Chain_Monte_Carlo

# G E N E R A T E  E X P E R I M E N T A L  S I G N A L
print("G E N E R A T E  E X P E R I M E N T A L  S I G N A L")

# Pick D and v points
print("Picking D and v points...")
num_knots = 2  # Ends of the profile

D = np.ones(num_knots)  # Currently have constant D and v profiles
v = np.ones(num_knots)

# Run STRAHL with D and v knots allowing for automatic STRAHL interpolation
# These should have the proper comments, input parameters, and summaries
# otherwise one should generate using the interface and then run!
print("Running STRAHL...")
main_fn = "./param_files/op12a_171122022_FeLBO3_experimental_signal"
inpt_fn = "test_input"
summ_fn = "test_summ.json"
if inpt_fn is None or summ_fn is None or main_fn is None:
    strahl.interface(main_fn,
                     inpt_fn=inpt_fn,
                     summ_fn=summ_fn,
                     verbosity=True)

# strahl.run(inpt_fn)

# Extract emmissivity, rho poloidal grid, time, and grid_points
# from STRAHL run
print("Extracting results from STRAHL...")
data_fn = "Festrahl_result.dat"

variables = ['diag_lines_radiation', 'rho_poloidal_grid',
             'time']
dimensions = ['grid_points']
attributes = []

groups = {"variables": variables,
          "dimensions": dimensions,
          "attributes": attributes}

results = strahl.extract_results(result=data_fn, **groups)

variables_data = results['variables']
dimensions_data = results['dimensions']
attributes_data = results['attributes']

emmissivity_ = variables_data['diag_lines_radiation']
emmissivity_raw = emmissivity_.data

rho_pol_grid_ = variables_data['rho_poloidal_grid']
rho_pol_grid = rho_pol_grid_.data
rho_pol_knots = [rho_pol_grid[0], rho_pol_grid[-1]]

time_ = variables_data['time']
time = time_.data

grid_points = dimensions_data['grid_points']

# Use the first charge state for our emissivity measurement
charge_state = 0

emmissivity = emmissivity_raw[:, charge_state, :]

# Integrate naively over the temporal dimension to get a profile
dimensions = emmissivity_.dimensions
integration_dimension = "time"

if integration_dimension in dimensions:
    integration_axis = dimensions.index(integration_dimension)
else:
    sys.exit("Exiting: Cannot find '{}' in '{}' dimensions."
          .format(integration_dimension, emmissivity_.name))

emmissivity_profile = np.sum(emmissivity, axis=integration_axis)

# Scale the profile by some factor
scale = 100000
scaled_emissivity_profile = np.multiply(scale, emmissivity_profile)

# Add noise to the profile
noise_factor = 2
sigma = noise_factor * np.sqrt(scaled_emissivity_profile)
noise_scaled_emissivity_profile = scaled_emissivity_profile + sigma

# plt.plot(rho_pol_grid, emmissivity_profile,
#          rho_pol_grid, noise_scaled_emissivity_profile, 'x-',
#          rho_pol_grid, scaled_emissivity_profile, 'x-')

# plt.legend(['Standard', 'Noisy', 'Scaled'])
# plt.show()


# Pick init guess for knots
D_knots = np.ones(num_knots)
v_knots = np.ones(num_knots)

# Initialize spline objects for both D and v
D_spline_ = Splines.linear_univariate_spline(rho_pol_knots, D_knots)
v_spline_ = Splines.linear_univariate_spline(rho_pol_knots, v_knots)

# F I T T I N G  L O O P
print("F I T T I N G  L O O P")

residual_ = Residual.strahl(D_spline_, v_spline_, verbose=False)
print(residual_)

res_keys = {'x': rho_pol_grid,
            'y': noise_scaled_emissivity_profile,
            'sigma': sigma
            }

coeffs = [10, 10, 4, 5]
r = residual_.mpfit_residual(coeffs,
                             x=rho_pol_grid,
                             y=noise_scaled_emissivity_profile,
                             sigma=sigma)


mdl = Least_Square(x=rho_pol_grid, y=noise_scaled_emissivity_profile,
                   sigma=sigma, residual_=residual_, fun_=None)

mdl.fit(coeffs=coeffs)

# def 1d_grid(length, points):
#     """
#     Given a length this calculates the positions in the range of
#     [0, length] to evenly distribute all the points.

#     Returns:
#         A list of size points describing the distance from 0 for
#         each point.
#     """
#     distance = np.divide(length, points - 1)

#     grid_point = 0
#     while grid_point <= length:
#         grid_points.append(grid_point)

#         grid_point = grid_point + distance

#     return grid_points




