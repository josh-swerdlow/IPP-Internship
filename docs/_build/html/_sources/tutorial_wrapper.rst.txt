Tutorial
========

In this tutorial, an outline for running PyStrahl will be demonstrated. This tutorial will show two simple ways to execute STRAHL through the PyStrahl interface via command line or programmatically. Finally, this tutorial will only scratch the surface of the full functionality of PyStrahl, but will still get the final result as needed.

Interface
~~~~~~~~~

In PyStrahl, an interface has been implemented which can be run with little to no input from the user. The interface method is written into the PyStrahl.core.strahl module and can be run from the command line or a python script. This interface, given the file names of the commented parameter files will create the appropriate objects for the given file names, extract the commented parameters from the files, generate an input file, and optionally execute strahl with the given input file. It will generate an input file and summary file based on the name you give through the command line when prompted. In addition, the input file will have the parameter values as you specify through the command line when prompted.

.. code-block:: python

    # First a demonstration of the PyStrahl interface method
    # main_fn 'op12a_171122022_FeLBO3'

    main_fn='op12a_171122022_FeLBO3_test'
    PyStrahl.core.strahl.interface(main_fn='op12a_171122022_FeLBO3_test')

This above example will run the interface a just a single main parameter file.

Quick Input File
~~~~~~~~~~~~~~~~

Alternatively, one does not need to go through the trouble of scanning the parameter files for commented parameters if one already knows all the parameters which are to be commented. For example, if one would like to run multiple STRAHL runs on 3 different values of the same parameter. In addition, this method will have no interaction with the command line so one can queue up multiple runs and let the code go.

.. code-block:: python

    # For example, I know that 'op12a_171122022_FeLBO3_test' has
    # 6 parameters commented out.

    # "# of interpolation points"
    D_interp_pts = 2
    # "rho polodial grid for interpolation"
    D_rho_grid = np.linspace(0, 10, 100)
    # "D[m^2/s]"
    D = [0, 4.5]
    # "# of interpolation points"
    v_interp_pts = 2
    # "rho polodial grid for interpolation"
    v_rho_grid = np.linspace(0, 10, 100)
    # "v[m/s]"
    v = [1, 2.1]

    inputs = [D_interp_pts, D_rho_grid, D,
              v_interp_pts, v_rho_grid, v]

    inpt_fn = 'brand_new2'
    strahl.quick_input_file(main_fn='op12a_171122022_FeLBO3_test',
                            inpt_fn=inpt_fn,
                            inputs=inputs)

    strahl.run(inpt_fn)

In the code snippet we set all the values for the the six parameter which are commented out in the main parameter file 'op12a_171122022_FeLBO3_test'. We then use the method quick_input_file found in PyStrahl.core.strahl which will generate an input file with the given main parameter file name, input file name, and input values (in order). Be warned, as documented in the quick_input_file method, this method will overwrite or create the input file given to it. Finally, we use a simple run method found in PyStrahl.core.strahl to execute a strahl run.