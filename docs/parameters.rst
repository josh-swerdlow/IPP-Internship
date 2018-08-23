Parameter Objects
=================

**Summary**
~~~~~~~~~~~

Parameter objects are the most basic object in the STRAHL Wrapper. They represent the entire state of a parameter as represented in the parameter files.

            M A I N  I O N
cv#     background ion: atomic weight   charge
                            4.0           2.0


i.e.) Below is a snippet from a main parameter file with four parameters.




            M A I N  I O N
cv#    background ion:  atomic weight    charge
                            4.0           2.0
            G R I D - F I L E
cv    shot      index
      22022       0

If this were translated into Parameter objects, then we would have an atomic_weight, charge, shot, and index Parameter object. Each of these objects have a variety of attributes to describe their state. Important ones include name, value, and state.

*name*: The name represents the exact text that is used to name the parameter inside of the parameter file. This is used for searching in the parameter files and thus it must match exactly is in the parameter file. In addition, parameter names must be unique within given parameter files.

*value*: The value is simply the stored data of the parameter

*state*: The state is a boolean that represents whether or not the parameter has been commented out with a '#' or not.

Using the example above.

*atomic_weight*: name = 'atomic weight', value = 4.0, state = True

*charge*: name = 'charge', value = 2.0, state = True

*shot*: name = 'shot', value = 22022, state = False

*index*: name = 'index', value = 0, state = False

This object is largely abstracted away inside of the ParameterFile object.

**Current Status**
~~~~~~~~~~~~~~~~~~

Th Parameters object is *not* complete.

1. It does not contain every parameter which can be found in the parameter files. To add more simply create a classmethod at the end of the file for the parameter and then add this parameter to the parameter files parameters dictionary (instantiated in ParameterFile of files.py).

2. There are not help files for every parameter in the parameter files or even the parameters currently inside this code.

**Future Work**
~~~~~~~~~~~~~~~

1. Fix points 1 and 2 of Current Status.

2. Currently valid_vals and dtype are not instantiated attributes for Parameter objects because they have not been implemented. Their intended function is to provide as a check for the user when they input values. Within Parameter.change_value(), there could be a quick boolean check to ensure the correct class instance was given. As well, some parameters can only expect a few unchanging values (such as string inputs for interpolations). Inputs for these parameters would be checked against those few valid values to protect the user from unintended typos which would otherwise require a restart of the program or cause a crash.

**Known Bugs**
~~~~~~~~~~~~~~~

There are no known bugs.

----------------------------------------------------------------------

.. module:: PyStrahl.core.parameters

.. autoclass:: Parameter
    :members:
    :special-members: __init__

