# Searches parameter file for parameter states
# parameter state: whether something has been commented or not
# Go from cv to cv
# checks for #
# extract feats next to #
# returns: filetype object
from loadParamFiles import loadParamFiles
from editor import paramFileEditor


def searchParameterFile(paramFile):
    """
    Searches the entire parameter file to turn on and off the
    parameters which have been elected to be changed
    """

    editor = paramFileEditor(paramFile)
    for i in range(0, 3):
        # Search until the next cv#
        if editor.next_cv():
            line = editor.readLines()
            print("This is the line: {}".format(line))

            # Extract parameters
            print("Extracting params")
            editor.extractParameters(line)

    # Repeat


if __name__ == '__main__':
    # main, background = loadParamFiles("op12a_171122022_FeLBO3",
    #         "pp22022.2", "goemTest", "literalShit")
    main, background = loadParamFiles("test", "pp22022.2",
            "goemTest", "literalShit")
    searchParameterFile(main)




