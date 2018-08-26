.. _josh_log:

This will act as an internal log for all of my (Josh Swerdlow) work on PyStrahl.

Current Tasks:
--------------
    Fix plotting class
    Add MCMC model
    Add GP model


Logs:
-----
Fix plotting class <-- task name
    PyStrahl.utils.plot <-- This refers to where the following changes took place
        Current - Playing with creating a plotting object to store relevant meta data and generalize plotting
            for multiple sets of files. I wouldn't touch this entire file, sorry. Let me know if this causes
            problems.

        8/24/18 - Added 'fn' optional argument to all methods. If a str value then the figure will be saved
            to that ./plot/'fn'. Otherwise it is not saved

            ^^ Added as a date because it is complete, otherwise it should be marked as current

    PyStrahl.utils.math <-- So we can make changes in different files, but all for one specific task!
        9/25/18 - Did some other stuff in math module that is relevant to this task :)

Add MCMC model
    PyStrahl.analysis.models.Markov_Chain_Monte_Carlo
        8/20/18 - Added plotting methods (plot_trace, plot_forest, plot_posterior, and plot_fit)

Add GP model
    ^^ There is nothing here, this just means I haven't done pushed anything to my remote that's worthwhile


Log History:
------------
Create Demo
    PyStrahl.demo
        8/24/18 - Completed the demo just in time for the presentation!

Implement Least Square Regression
    PyStrahl.analysis.models.Least_Square
        8/20/18 - Completed work and finalized Least Sq Regression
        8/16/18 - Some more stuff
        8/15/18 - Some stuff I did
