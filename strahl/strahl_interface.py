# This will incorporate interacting with the user, fixing the files,
# and finally executing the strahl file
from loadParamFiles import loadParamFiles
from strahl_run import strahl_run


def main():
    # load parameterFiles [done]
    mainFileName = ""
    bckgrndFileName = ""
    geomFileName = ""
    fluxFileName = ""
    mainFile, backgroundFile = loadParamFiles(mainFileName,
        bckgrndFileName, geomFileName, fluxFileName)

    # extract parameters [working]


    # Ask user to load or create new input files
    # Create

    # Load

    # Clean-up

    # Execute strahl
    strahl_run()


if __name__ == '__main__':
    main()
