STRAHL Wrapper
==============

The purpose of this code is to generate a nice, optimized, and user friendly
interface to using the STRAHL software.

At the highest level, the code leverages STRAHL's built in commenting out syntax ('cv#') in parameter files. By commenting out a parameter from a parameter file, STRAHL then waits for command line input of the parameter or reads from an input file using file redirecting from the command line ('<').

This code uses this feature to take already commented parameter files and generate input files for them. It generates input files by either taking the values for the commented parameters in programmatically or through the command line. When the command line is used in this code (as opposed to STRAHLs command line input), there is significantly more user interaction to ensure the correct values are given to the input file. Once these input files are generated, summary files are also created which
can store the entire state of a STRAHL run (the parameter file and input file states) in JSON formatting. This gives the user a summary of what input file they have just created and for what parameter files.

.. toctree::
    :maxdepth: 2
    :caption: Directory

    getting_started_wrapper
    tutorial_wrapper
    parameters
    file
    editor
    query
    auxillary
