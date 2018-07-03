from parameterFile import parameterFile
import subprocess


class FileEditor():

    def __init__(self, file):
        self.fileName = file
        self.file = FileLineWrapper(open(file, "r+", 1))
        self.line = 0

    def comment(self):
        """
        If something is commented in the parameter files, then
        it should be immediately added to the input file
        """
        pass

    def uncomment(self):
        """
        If something is uncommented in the parameter files, then
        it should be immediately removed from the input file
        """
        pass

    def add(self, addition):
        self.file.write(addition)

    def delete(self, deletion=None):
        if deletion is None:  # delete the current line
            line = self.file.line
            print("sed {}d {}".format(line, self.fileName))
            subprocess.run("sed {}d {}".format(line, self.fileName))
            # subprocess.run(["sed \'{}d\' {}".format(line, self.fileName)])
        else:  # delete the deletion
            subprocess.run(["sed \'{}d\' {}".format(deletion, self.fileName)])


class quickFileEditor(FileEditor):

    def __init__(self, paramFile, inputFile=None):

        self.main = parameterFile.mainFile()
        self.background = parameterFile.backgroundFile()
        self.geometry = parameterFile.geometryFile()
        self.flux = parameterFile.fluxFile()

        self.total_parameters = (self.main.size + self.background.size +
                            self.geometry.size + self.flux.size)

        self.paramFile = open(paramFile, "w+", 1)

        # beware that the number of line in a premade input file
        # and a new set of parameterFiles could cause hierarchy
        # problems
        if inputFile is None:
            inputFile = self.temp_file(self.total_parameters)
        else:
            inputFile = open(inputFile, "w+", 1)
        self.inputFile = inputFile

    @staticmethod
    def temp_file(total_parameters):

        tmp = open("temp", 'w+', 1)

        tmp.write("Main File Name\n")
        tmp.write("0.00\n")

        for params in range(0, total_parameters):
            tmp.write("\n")

        tmp.write("E\n")

        return tmp


class FileLineWrapper(object):

    def __init__(self, f):
        self.f = f
        self.line = 0

    def close(self):
        return self.f.close()

    def readline(self):
        self.line += 1
        return self.f.readline()

    def readlines(self):
        return self.f.readlines()

    def write(self, text):
        return self.f.write(text)

    def reset_position(self):
        self.f.seek(0)

    # to allow using in 'with' statements
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



