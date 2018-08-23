ParameterFile Object
====================

**Summary**
~~~~~~~~~~~

This object ecapsulates the entire state of a given parameter file.

A parameter file is defined as a main, background, geometry, flux,
or atomic data file which has the parameter values for STRAHL to read.

Each file has a different file name syntax and storage location. Please
see the STRAHL manual_ for more information.
    * ./param: main files, atomic data
    * ./nete: background, geometry, flux

.. _manual: http://pubman.mpdl.mpg.de/pubman/item/escidoc:2143869/component/escidoc:2143868/IPP_10_30.pdf

**Current Status**
~~~~~~~~~~~~~~~~~~

The ParameterFile object is *not* fully operational.

1. The ParameterFile object has no support implemented for background, geometry, flux, or atomic data parameter files.

**Future Work**
~~~~~~~~~~~~~~~

There is no future work planned at this moment. Point 1 of Current Status is a low priority and can be implemented by request. Unfortunately, my research focus around varying D and v profiles which are parameters in the main file only. As such, I have not had time to implement the support for other parameter files.

**Known Bugs**
~~~~~~~~~~~~~~~

There are no known bugs.


.. module:: ..PyStrahl.core.file

.. autoclass:: ParameterFile
   :members:
   :special-members: __init__