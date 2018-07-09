from parameterFile import parameterFile
import subprocess
import os
import re

class FileEditor():

    def __init__(self, file):
        self.fileName = file
        self._deletions = 0
        self._line = 0

        if not os.path.isdir("./_backup"):
            mkdirBckupCmd = "mkdir ./_backup"
            subprocess.call(mkdirBckupCmd.split())

    def openFile(self, read_write_attribute="r+", buffer_time=1):

        self.file = open(self.fileName, read_write_attribute, buffer_time)

    def readChar(self, n=1):
        """
        Returns n characters read from the buffer
        and keeps track of the current line.
        This should be used to capture select words
        from a setence, rather than read line or lines.
        For that use FileEditor.readline()
        """
        if self.file.readable():
            chars = self.file.read(n)
            for char in chars:
                if char is "\n":
                    self._line += 1
            return chars
        else:
            print("File was not opened to be readable.")
            raise OSError

    def readLines(self, n=1):
        """
        Returns the line read from the buffer
        and keeps track of the current line.
        """
        if self.file.readable():
            self._line += n
            lines = str()
            for l in range(0, n):
                lines = lines + self.file.readline()
            return lines
        else:
            print("File was not opened to be readable.")
            raise OSError

    def curLoc(self):
        """
        Returns the current location in the form of
        the current character and the current line.
        """
        if self.file.seekable():
            return (self.file.tell(), self._line)
        else:
            print("File cannot be randomly accessed.")
            raise OSError

    def move(self, loc=(0, 0)):
        """
        Resets the buffer to the specified location
        relative to the begining of the buffer and
        resets the line attribute to the line at the
        reset location.
        """
        charNumb = loc[0]
        lineNumb = loc[1]
        if self.file.seekable():
            self.file.seek(charNumb, 0)
            self._line = lineNumb
        else:
            print("File cannot be randomly accessed.")
            raise OSError

    def peak(self, n=1):
        """
        Reads n characters read from the buffer,
        but returns to the initial location.
        """
        loc = self.curLoc()
        chars = self.readChar(n)

        self.move(loc)

        return chars

    def peakLines(self, n=1):
        """
        Reads until a newline is found or EOF,
        but returns to the initial location.
        """
        loc = self.curLoc()
        line = self.readLines(n)

        self.move(loc)

        return line

    def update(self):
        self.file.flush()

    def add(self, addition):
        if self.file.writable():
            self.file.write(addition)
            self._line += addition.count("")

        else:
            print("File is not writable.")
            raise OSError

    def delete(self, deletion="", replacement="", line=0, repeat=1, backup=False):
        """
        Deletes the given deletion phrase or by default
        a single line. The previous file is always saved
        into a folder called ./_backup.
        """
        empty = ""
        self._deletions += 1

        if deletion is empty:
            delLineCmd = ("sed -l -i.bck_up{} {}d {}"
                .format(self._deletions, self._line + 1, self.fileName))
            subprocess.call(delLineCmd.split())
            self.update()
        elif deletion is not empty:  # delete the deletion
            if line is 0:
                line = self._line + 1
            delPhraseCmd = ("sed -l -i.bck_up{} {}s/{}/{}/{} {}"
                .format(self._deletions, line, deletion,
                        replacement, repeat, self.fileName))
            print(delPhraseCmd + "\n")
            subprocess.run(delPhraseCmd.split())
            self.update()
        else:
            print("Broke af")

        if backup:
            mvBckupCmd = ("mv {}.bck_up{} ./_backup"
                .format(self.fileName, self._deletions))
            subprocess.call(mvBckupCmd.split())
        else:
            delBckupCmdn = (("rm {}.bck_up{}")
                .format(self.fileName, self._deletions))
            subprocess.call(delBckupCmdn.split())

    def closeFile(self, clean=1):
        """
        Closes the file and by default deletes the
        backups for this file in the folder ./_backup
        """
        if clean:
            cleanBckupDir = "rm ./_backup/{}.bck_up*".format(self.fileName)
            subprocess.call(cleanBckupDir.split())
        self.file.close()


class paramFileEditor(FileEditor):
    """
    A class that handles general editing of parameter files.
    It is built on top of the FileEditor() class.
    Needs to deal with comments and stuff
    """

    def __init__(self, paramFile):
        super().__init__(paramFile.fn)
        self.paramFile = paramFile
        self.openFile()

    def next(self, regex):
        """
        Moves to the first next regex in the given file
        """
        while self.peakLines() != '':
            line = self.peakLines()
            if re.match(regex, line) is not None:
                return 1
                break
            else:
                self.readLines()
                return -1

    def next_cv(self):
        """
        Moves to the next 'cv ' in the given file.
        """
        return self.next('(cv\\s*)')

    def next_cvPound(self):
        """
        Moves to the next 'cv#' in the given file
        """
        return self.next("(cv#)")

    def extractParameters(self, line):
        """
        Given a parameter header input, this extracts the parameters
        """
        # Move to the first non-space character after the 'cv '
        splitTemp = re.split("cv#\\s*", line)
        print(splitTemp)
        temp = ""
        if len(splitTemp) == 2:
            temp = splitTemp[1][0:-1]  # Remove the newline character
            print(temp)
        else:
            return -1
        params = re.split("\\s{2,}", temp)



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


class inputFileEditor():
    """
    """
    pass


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



