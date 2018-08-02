########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: Searches parameter file to change parameter
    states as requested.

To Do:

"""

__author__ = 'Joshua Swerdow'

from loadParamFiles import loadParamFiles
from editor import ParameterFileEditor


def searchParameterFiles(paramFiles, searchParam="comments", verbose=False):
    """
    Iterates over a series of ParameterFile objects and searches for
    the given search parameter in each.

    Parameters:
        * **paramFiles** [list]: list of ParameterFile objects to iterate over
        * **searchParam** [str|"comments"]: chooses to search for comments of
            non-comments
        * **verbose** [bool|False]: turn on commented output
    """
    for file in paramFiles:
        if verbose:
            print("Searching {}".format(file))

        searchParameterFile(file, searchParam)


def searchParameterFile(paramFile, searchParam="comments", verbose=False):
    """
    Searches the entire parameter file to turn on and off the
    parameters which have been elected to be changed with comments ('#').

    Parameters:
        * **paramFile** [ParameterFile obj]: a ParameterFile object
            to search through
        * **searchParam** [str|"comments"]: chooses to search for comments of
            non-comments
        * **verbose** [bool|False]: turn on commented output
    """
    # Initialize a ParameterFileEditor obj
    editor = ParameterFileEditor(paramFile.fn)

    # Search entire file for comments and non comments
    # then change the relevant parameters
    keywords = set()
    while keywords is not None:
        if searchParam == "noncomments":
            if verbose:
                print("Searching parameter file {} for non-comments"
                    .format(paramFile.fn))

            keywords = _searchParameterFileNoncomments(editor)
        else:
            if verbose:
                print("Searching parameter file {} for comments"
                    .format(paramFile.fn))

            keywords = _searchParameterFileComments(editor)

        # Change the relevant parameters if there are any
        if keywords is not None and len(keywords) != 0:
            _changeRelevantParameters(paramFile, keywords)


def _searchParameterFileComments(editor):
    """
    Given an ParameterFileEditor object go to the next comment
    line, extract the commented parameters from that line, and
    return them in a set of keywords.

    Parameters:
        * **editor** [ParameterFileEditor obj]: the parameterFile objects
            editor object

    Returns:
        * **keywords** [set]: A set of keywords found in the
            parameter file
    """
    keywords = set()

    next_comment = editor.next_comment()

    if next_comment is None:
        keywords = None

    elif next_comment:
        line = editor.readLines()

        keywords = editor.extractCommentedParameters(line)

    return keywords


def _searchParameterFileNoncomments(editor):
    """
    Given an ParameterFileEditor object go to the next non-comment
    line, extract the non-commented parameters from that line, and
    return them in a set of keywords.

    Parameters:
        * **editor** [ParameterFileEditor obj]: the parameterFile objects
            editor object

    Returns:
        * **keywords** [set]: A set of keywords found in the
            parameter file
    """
    keywords = set()

    next_comment = editor.next_noncomment()

    if next_comment is None:
        keywords = None

    elif next_comment:
        line = editor.readLines()

        keywords = editor.extractNoncommentedParameters(line)

    return keywords


def _changeRelevantParameters(paramFile, keywords):
    """
    Each paramFile obj is created for the different parameterFiles
    (i.e. main, background, geometry, and flux). Each paramFile obj
    contains many attributes. One is a dict of parameter objects.
    This parameter objects contain the name, value, state, and other
    importan attributes of the parameter. The value and state can
    be changed through the paramter object. The name is initiated
    on creation of the object.

    Searching the parameter file currently yields a list of
    parameter keywords taken from the header line. However,
    one look inside the parameter files can show that not all of
    these keywords are actually parameters which require input.
    So this function cross-references the paramFile params dict
    and the params list from searching. It also changes their
    state while we have the relevant parameters.
    """
    params = set(paramFile.paramDict.keys())
    relevParams = params & keywords

    # Change parameters to correct state
    [paramFile.paramDict[paramName].changeState() for paramName in relevParams]


if __name__ == '__main__':
    main, background = loadParamFiles("op12a_171122022_FeLBO3",
            "pp22022.2", "goemTest", "literalShit")
    # main, background = loadParamFiles("test", "pp22022.2",
    #         "goemTest", "literalShit")
    searchParameterFile(main)




