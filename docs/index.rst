.. strahl documentation master file, created by
   sphinx-quickstart on Sun Aug 19 23:34:05 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyStrahl's Documentation!
=====================================

IPP Internship Summer 2018
--------------------------
For the summer of 2018, I will be working at the Max Plank Institute for Plasma Physics also known as Institue for Plasma Physics (IPP). In this internship, I will be working with Professor Novimir Pablant (Princeton Plasma Physics Laboratory) on an investigation on the resolution of *STRAHL* fitting capabilities. I will compare the Least Square (LSTSQ) regression algorithm and capabilities already implemented within STRAHL with Markov Chain Monte Carlo (MCMC), Gaussian Process (GP), and a combined MCMC-GP regression algorithm. While doing this research, I will also work closely with the STRAHL simulation software and look to design and optimized workflow. While doing all of this, I will maintain a BitBucket repository, working documentation of my code, and reports on my findings.

Goals
-----
1. Primary Goals:
    a. Rework STRAHL workflow to be faster and easier to use.
    b. Measure goodness of fit and speed of the following algorithms in order to understand resolution of D and V profiles.
        * Least Squares (LSTSQ) regression
        * Markov Chain Monte Carlo (MCMC) regression
        * Gaussian Process (GP) regression
        * [Possibly] Combined MCMC GP regression

2. Seconday Goals:
    a. Understand how to use GitHub/BitBucket
    b. Create good documentation on code
    c. Maintain reports of findings

3. Optional Goals:
    a. Do something with the data base

Documentation
-------------
.. toctree::
   :maxdepth: 2

   strahl
   analysis

Built With:
~~~~~~~~~~~
* PyMC3_ - bayesian statistics package
* mirmpfit_ -  non-linear least squares curve fitting package

.. _PyMC3: https://docs.pymc.io/
.. _mirmpfit: https://bitbucket.org/amicitas/mirmpfit/src/master/

Authors
~~~~~~~
* **Joshua Swerdlow** - Github_ & BitBucket_

.. _GitHub: https://github.com/josh-swerdlow
.. _BitBucket: https://bitbucket.org/josh-swerdlow/

Acknowledgments
~~~~~~~~~~~~~~~
* Yale
* IPP
* Peoples



