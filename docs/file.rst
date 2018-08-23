File Objects
============

There are three types of File objects in this code: ParameterFile, InputFile, and SummaryFile. Each one of them represents the state of the actual file they represent.
In the case of ParameterFile, it can represent a main file, geometry file, or background file depending on initialization.

The File object stores the data structures that describe the state of the actual file; as well, as storing methods that create, interact, or load the actual file. For example, a ParameterFile object will have a dictionary of Parameter objects key with the Parameter object's name. An InputFile object will have a list of inputs that are expected to be placed in the input file. In all cases the object will have file name attributes (fn), and sometimes other metadata to describe the object.


.. toctree::
    :maxdepth: 2
    :caption: Classes within file.py

    parameter_file
    input_file
    summary_file


