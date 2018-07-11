# This will incorporate interacting with the user, fixing the files,
# and finally executing the strahl file
from loadParamFiles import loadParamFiles
from searchParameterFile import searchParameterFile
from strahl_run import strahl_run


def main():
    # load parameterFiles [done]
    # TODO: Give interface optional command line arguments for
    #   verbosity
    #   fileNames
    #   Create or Load input file
    #   etc
    mainFileName = "op12a_171122022_FeLBO3"
    backgroundFileName = "pp22022.2"
    geomFileName = ""
    fluxFileName = ""
    mainFile, backgroundFile = loadParamFiles(mainFileName,
        backgroundFileName, geomFileName, fluxFileName)

    # extract parameters [done]
    searchParameterFile(mainFile)
    searchParameterFile(backgroundFile)

    # Ask user to load or create new input files [working]

    # Create

    # Load

    # Clean-up

    # Execute strahl
    strahl_run()


if __name__ == '__main__':
    main()
