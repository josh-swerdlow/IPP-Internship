# Searches parameter file to change parameter states as requested
from loadParamFiles import loadParamFiles
from editor import parameterFileEditor


def searchParameterFiles(paramFiles, searchParam="comments"):
    for file in paramFiles:
        searchParameterFile(file, searchParam)


def searchParameterFile(paramFile, searchParam="comments"):
    """
    Searches the entire parameter file to turn on and off the
    parameters which have been elected to be changed

    Alternatively can search the entire parameter file for parameters
    which have not been elected to be changed. This will aid in
    any future implementation in which we would like to turn these
    on and off interactively.
    """
    editor = parameterFileEditor(paramFile.fn)

    keywords = set()
    while keywords is not None:
        if searchParam == "noncomments":
            # print("Searching parameter file {} for non-comments"
            #     .format(paramFile.fn))
            keywords = searchParameterFileNoncomments(editor)
        else:
            # print("Searching parameter file {} for comments"
            #     .format(paramFile.fn))
            keywords = searchParameterFileComments(editor)

        if keywords is not None and len(keywords) != 0:
            changeRelevantParameters(paramFile, keywords)


def searchParameterFileComments(editor):
    keywords = set()
    next = editor.next_comment()

    if next is None:
        keywords = None
    elif next:
        line = editor.readLines()
        keywords = editor.extractCommentedParameters(line)

    return keywords


def searchParameterFileNoncomments(editor):
    keywords = set()
    next = editor.next_noncomment()

    if next is None:
        keywords = None
    elif next:
        line = editor.readLines()
        keywords = editor.extractNoncommentedParameters(line)

    return keywords


def changeRelevantParameters(paramFile, keywords):
    """
    Each paramFile obj is created for the different parameterFiles
    (i.e. main, background, geometry, and flux). Each paramFile obj
    contains many attributes. One is a list of parameter objects.
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




