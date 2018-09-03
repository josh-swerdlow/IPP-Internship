Getting Started
===============

STRAHL
------

For more information about STRAHL see the manual_.

.. _manual: http://pubman.mpdl.mpg.de/pubman/item/escidoc:2143869/component/escidoc:2143868/IPP_10_30.pdf

File Dependencies
-----------------
There are numerous file dependencies to have this STRAHL Wrapper run correctly. For reference, the STRAHL directory (or main directory '.') is the directory in which one would run STRAHL from (./strahl).

*Source Code*:
The source code for the wrapper can be found in IPP Internship/PyStrahl/core and should be put in ./source/PyStrahl/core. This includes source code for all parameter, file, and editor objects.

*Parameter Files*:
As with STRAHL main parameter files should be put inside ./param_files and the remaining parameter files (background, geometry, etc) should be put inside ./nete.
The STRAHL wrapper will be looking for these files in these specific locations and if they have been changed then one must also change them in the wrappers code. I do not suggest this.

*Result Files*:
As with STRAHL, results are stored as netCDF files in ./result.
The STRAHL wrapper will be looking for these files in this specific locations and if it has been changed then one must also change it in the wrappers code. I do not suggest this.

*Help Files*:
For addition functionality a IPP Internship/PyStrahl/core/help directory contains <parameter_name>.h files. These help files contain important documentation about the specific parameter. When Parameter objects are created, they will automatically be given a help attribute which contains the entire help file (if it can be found). These help files should be put into ./source/PyStrahl/core/help.

*Input Files*:
Input files should be left in the main directory '.' as they are called as:
./strahl < inpt_fn

*Summary Files*:
Summary files are found inside IPP Internship/summaries and so a ./summaries directory should be created to store these.

*Plot Figures and Graphics*:
Plots are found inside IPP Internship/plots and so a ./plots directory should be create to store these.

Packages
--------
* scipy_
* numpy_
* termcolor_

.. _scipy: https://www.scipy.org/
.. _numpy: http://www.numpy.org/
.. _termcolor: https://pypi.org/project/termcolor/
Current Status
--------------
The current status of this wrapper can be summarized as working as intended, but with limitations. As the project moved forward, I began to realize the specific functionality that this very general architecture would have to fullfill. As such, more specific functionality has been added in places as simple part time fixes. However, I will continue to work on the analysis side of my project and maintain, clean, and upgrade my code for the foreseeable future. Please do not hesitate to reach out for question, concerns, or bugs.

Future Work
-----------
In addition to maintaining the code and making adjustments or changes by request of the users there are two large undertakings I have in mind.


1. I would like to strip away much of the user interaction with the command line. As much as command line interface can be intuitive and a good way to begin working with STRAHL, it is not a good way to run numerous tests. So, I would like to develop an API that works directly with the netCDF files that STRAHL generates for each run from the parameter files and input file. Ideally, I would like to eliminate the need for STRAHL to read the parameter files. This will require some rewriting of the STRAHL source code and so I would like create my own strahl_js.

2. I would like to scale this project such that one could run numerous STRAHL runs in parallel. Thereby allowing the user to test numerous different parameter settings simultaneously. This will likely be written in C or C++ and then wrapped into PyStrahl for use from Python.

Please note that these last two points are not my highest priority. I will be working with what I have (since it works) and looking to complete my analysis of different regression algorithms. Only if it is convenient, I find myself with extra time, or it seems extra-ordinarily useful to more than just myself will these be implemented.