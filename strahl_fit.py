# import sys
# sys.path.append("./source")

# import matplotlib
# matplotlib.use('PDF')

import numpy as np
import matplotlib.pyplot as plt

from PyStrahl.utils import math
from PyStrahl.core import strahl, editor
from PyStrahl.analysis.models import Least_Square
from PyStrahl.analysis.data import Splines, Residual


def strahl_fit_least_square_data(sum_fn):
    index = 0
    dim = 'grid_points'

    signal, rho_pol_grid, sigma = generate_experimental_signal(index, dim)

    summary_ = editor.SummaryFileEditor(sum_fn, verbosity=True)

    strahl_fit_least_square(summary_, signal, rho_pol_grid, sigma, index, dim)

    fit_info = summary_.get()

    print(fit_info)


# G E N E R A T E  E X P E R I M E N T A L  S I G N A L
def generate_experimental_signal(index, dim):

    print("G E N E R A T E  E X P E R I M E N T A L  S I G N A L")

    # Pick D and v points
    print("Picking D and v points...")
    num_knots = 2  # Ends of the profile

    # Constant D and v knots initially
    D_knots_init = np.asarray([0.1, 0.1])
    v_knots_init = np.asarray([0.1, 0.1])

    # Ends points of plasma initially
    x_knots_init = np.asarray([0.0, 1.0])

    inputs = [num_knots, x_knots_init, D_knots_init,
              num_knots, x_knots_init, v_knots_init]

    # Run STRAHL with D and v knots allowing for automatic STRAHL interpolation
    # These should have the proper comments, input parameters, and summaries
    # otherwise one should generate using the interface and then run!
    print("Running STRAHL...")
    main_fn = "op12a_171122022_FeLBO3_test"
    inpt_fn = "test_input"

    strahl.quick_input_file(main_fn=main_fn, inpt_fn=inpt_fn, inputs=inputs,
                            verbose=False)

    strahl.run(inpt_fn, verbose=False)

    # Extract emmissivity, rho poloidal grid, time, and grid_points from STRAHL run
    print("Extracting results from STRAHL...")
    data_fn = "Festrahl_result.dat"

    # Lists of names in the netCDF file
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

    emissivity_ = variables_data['diag_lines_radiation']

    rho_pol_grid_ = variables_data['rho_poloidal_grid']
    rho_pol_grid = rho_pol_grid_.data

    time_ = variables_data['time']
    time = time_.data

    grid_points = dimensions_data['grid_points']

    # Select a charge state, integrate naively over the selected dimension,
    # scale the profile by some factor, add scaled noise to the signal
    print("Generating signal...")

    signal_scale, noise_scale, background_scale = 100000, 2, 10

    signal_outputs = math.generate_signal(emissivity_,
                                          charge_index=index,
                                          integrat_dim=dim,
                                          signal_scale=signal_scale,
                                          noise_scale=noise_scale,
                                          background_scale=background_scale)

    profile, scaled_profile, signal, sigma = signal_outputs

    plt.plot(time, profile,
             time, signal, 'x-',
             time, scaled_profile, 'x-')

    plt.legend(['Standard', 'Noisy', 'Scaled'])
    plt.xlabel('Time in {}'.format(time_.units.decode()))
    plt.title('Simulated Experimental Signal')
    plt.xlim([2.0, 2.5])

    plt.savefig('./plots/initial_profiles.pdf')

    return signal, rho_pol_grid, sigma


# F I T T I N G  L O O P
def strahl_fit_least_square(summary_, signal, rho_pol_grid, sigma, index, dim):
    print("F I T T I N G  L O O P")
    rho_pol_knots = [rho_pol_grid[0], rho_pol_grid[-1]]

    # Make some guesses about D and v knot locations
    D0_guess = 10
    D1_guess = 10
    v0_guess = 4
    v1_guess = 5

    D_guess = [D0_guess, D1_guess]
    v_guess = [v0_guess, v1_guess]

    # Initialize spline objects for both D and v guesses
    D_spline_ = Splines.linear_univariate_spline(rho_pol_knots, D_guess)
    v_spline_ = Splines.linear_univariate_spline(rho_pol_knots, v_guess)

    # Establish file names
    main_fn = "op12a_171122022_FeLBO3_test"
    inpt_fn = "fitting_loop_input"
    data_fn = "Festrahl_result.dat"

    residual_ = Residual.strahl(D_spline_, v_spline_, main_fn, inpt_fn,
                                data_fn, index, dim,
                                strahl_verbose=False, verbose=False)

    # Make a guess about the scale factor
    scale_guess = 1.8
    background_guess = 5

    parnames = ['D0', 'D1', 'v0', 'v1', 'noise', 'background']
    guesses = [D0_guess, D1_guess, v0_guess, v1_guess,
               scale_guess, background_guess]

    # Initialize some constants
    STEP = 1.0e-4
    D_LOWER_BOUND = 5.0e-3
    SCALE_LOWER_BOUND = 0.0e0
    BACK_LOWER_BOUND = 0.0e0
    BACK_UPPER_BOUND = 100e0

    # Instantiate parinfo list
    parinfo = list()
    for parname, guess in zip(parnames, guesses):
        dic = dict()

        dic['fixed'] = True
        dic['parname'] = parname
        dic['step'] = STEP
        dic['limited'] = [False, False]
        dic['limits'] = [0.0, 0.0]

        if parname.startswith("D"):
            dic['limited'] = [True, False]
            dic['limits'] = [D_LOWER_BOUND, 0.0]

        elif parname is "scale":
            dic['limited'] = [True, False]
            dic['limits'] = [SCALE_LOWER_BOUND, 0.0]

        elif parname is "background":
            dic['limited'] = [True, True]
            dic['limits'] = [BACK_LOWER_BOUND, BACK_UPPER_BOUND]

        parinfo.append(dic)

    # Change all parameters to 'unfixed'
    for info in parinfo:
        info['fixed'] = False

    mdl = Least_Square(rho_pol_grid, signal, sigma, parinfo, residual_,
                       verbose=False)

    mdl.fit(coeffs=guesses)

    summary_.write(mdl.__dict__)

