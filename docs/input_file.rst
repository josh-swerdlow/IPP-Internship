InputFile Object
================

**Summary**
~~~~~~~~~~~

This object ecapsulates the entire state of an input file.

The actual input file is defined as a list of inputs that STRAHL expects to read in
from the parameter files in the order STRAHL expects to receive them. This object handles the creation of an InputFile object by either creating or loading (and overwriting) an input file in the main directory.

Input files are store in the same directory the STRAHL is executed from '.' so that they may be called as ./strahl < inpt_fn.

**Current Status**
~~~~~~~~~~~~~~~~~~

The InputFile object is fully operational.

**Future Work**
~~~~~~~~~~~~~~~

There is no future work planned at this moment.

**Known Bugs**
~~~~~~~~~~~~~~~

There are no known bugs.

------------------------------------------------------------------------------------

**Source Code**
~~~~~~~~~~~~~~~

.. module:: PyStrahl.core.file

.. autoclass:: InputFile
    :members:

