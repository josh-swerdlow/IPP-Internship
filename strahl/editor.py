from auxillary import generateDictionary
from file import inputFile
from file import summaryFile

import subprocess as sub
import numpy as np

import json
import sys
import re
import os


class FileEditor():

    def __init__(self, file):
        self.fileName = file
        self._deletions = 0
        self._line = 0

        # if not os.path.isdir("./_backup"):
        #     mkdirBckupCmd = "mkdir ./_backup"
        #     sub.call(mkdirBckupCmd.split())

    def openFile(self, read_write_attribute="r+", buffer_time=1):

        self.file = open(self.fileName, read_write_attribute)

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
            if isinstance(addition, np.ndarray):
                addition.tofile(self.file, sep=" ", format="%.1f")
                self.file.write("\n")

                self._line += 1
            else:
                self.file.write(addition)
                self._line += addition.count("\n")

        else:
            print("File is not writable.")
            raise OSError

    def delete(self, deletion="", replacement="", line=0, repeat=1):
        """
        Deletes the given deletion phrase or by default
        a single line. The previous file is always saved
        into a folder called ./_backup.
        """
        empty = ""
        self._deletions += 1
        if deletion is empty:
            print(self.curLoc())
            # print("Deleting: " + self.peakLines())
            delLineCmd = ("sed -l -i.bck_up{} {}d {}"
                .format(self._deletions, self._line + 1, self.fileName))

            print(delLineCmd + "\n")
            sub.call(delLineCmd.split())

            loc = self.curLoc()

        elif deletion is not empty:  # delete the deletion
            if line is 0:
                line = self._line + 1

            delPhraseCmd = ("sed -l -i.bck_up{} {}s/{}/{}/{} {}"
                .format(self._deletions, line, deletion,
                        replacement, repeat, self.fileName))

            print(delPhraseCmd + "\n")

            sub.run(delPhraseCmd.split())

            loc = self.curLoc()

        else:
            print("Broke af")

        self.file.close()
        self.openFile()
        self.move(loc)

        # if backup:
        #     mvBckupCmd = ("mv {}.bck_up{} ./_backup"
        #         .format(self.fileName, self._deletions))
        #     sub.call(mvBckupCmd.split())
        # else:
        #     delBckupCmdn = (("rm {}.bck_up{}")
        #         .format(self.fileName, self._deletions))
        #     sub.call(delBckupCmdn.split())

    def closeFile(self, clean=True):
        """
        Closes the file and by default deletes the
        backups for this file in the folder ./_backup
        """
        if clean:
            cleanBckupDir = "rm ./_backup/{}.bck_up*".format(self.fileName)
            sub.call(cleanBckupDir.split())
        self.file.close()


class parameterFileEditor(FileEditor):
    """
    A class that handles general editing of parameter files.
    It is built on top of the FileEditor() class.
    Needs to deal with comments and stuff
    """

    def __init__(self, paramFileName):
        super().__init__(paramFileName)
        self.openFile()

    def next(self, regex):
        """
        Moves to the first next regex in the given file
        """
        while self.peakLines() != '':
            line = self.peakLines()
            if re.match(regex, line) is not None:
                return True
                break
            else:
                self.readLines()
                return False

        return None

    def extractParameters(self, line, regex):
        """
        Given a parameter header input, this extracts the parameters
        """
        splitTemp = re.split(regex, line)

        temp = ""
        if len(splitTemp) == 2:
            temp = splitTemp[1][0:-1]  # Remove the newline character
            # print("2. temp: {}".format(temp))
        else:
            print("The parameter header was not split correclty.")
            return -1

        paramsSet = set(re.split("\\s{2,}", temp))

        return paramsSet

    def next_noncomment(self):
        """
        Moves to the next 'cv ' or 'cv#' in the given file.
        """
        return self.next("(cv\\s){1}")

    def next_comment(self):
        """
        Moves to the next 'cv ' or 'cv#' in the given file.
        """
        return self.next("(cv#)")

    def extractNoncommentedParameters(self, line):
        """
        Given a parameter header input, this extracts the
        uncommented parameters
        """
        return self.extractParameters(line, "cv\\s*")

    def extractCommentedParameters(self, line):
        """
        Given a parameter header input, this extracts the
        commented parameters
        """
        return self.extractParameters(line, "cv#\\s*")

    # def comment(self):
    #     """
    #     If something is commented in the parameter files, then
    #     it should be immediately added to the input file
    #     """
    #     pass

    # def uncomment(self):
    #     """
    #     If something is uncommented in the parameter files, then
    #     it should be immediately removed from the input file
    #     """
    #     pass


class summaryFileEditor(summaryFile):  # [todo]

    def __init__(self, sum_fn):
        super().__init__(sum_fn)
        self.loadable = False

    def get(self):
        """
        Gets the summary file
        """
        obj = None

        path_to_file = os.path.join(self.path, self.fn)

        if self.serial == "json":
            obj = self._getJSON(path_to_file)

        elif self.serial == "hdf5":
            obj = self._getHDF5(path_to_file)

        else:
            print("Incorrect serial value: {}".format(self.serial))

        return obj

    def write(self, dic):
        """
        Writes the input file into a summary file
        """
        path_to_file = os.path.join(self.path, self.fn)

        if self.serial == "json":
            self._writeJSON(dic, path_to_file)

        elif self.serial == "hdf5":
            self._writeHDF5(dic, path_to_file)

        print("Writing {} to {}".format(dic, path_to_file))

        self.loadable = True

    @staticmethod
    def _getJSON(fn):
        obj = None

        with open(fn, "r") as f:
            obj = json.load(f)

        return obj

    @staticmethod
    def _writeJSON(dic, fn):
        with open(fn, "w") as f:
            json.dump(dic, f, indent=4)

    @staticmethod
    def _getHDF5(obj):
        pass

    @staticmethod
    def _writeHDF5(obj):
        pass


class inputFileEditor(inputFile):  # [check]

    def __init__(self, inpt_fn=None, inputs=None):  # [check]
        super().__init__(inpt_fn, inputs)

    def add(self, element=None, index=None):  # [check]
        """
        Adds a value (if given) to the input file
        if it is not given, then the user is asked to
        input a value
        """
        if element is None:
            print("The follow request will be evaluated like python code!\n" +
                  "As such, don't forget to put \"\" around strings.\n" +
                  "In addition, the numpy (np) module can be used.")
            element = input("Please input a value: ")
            element = eval(element,
                          {'__builtins__': None, "np": np},
                          {'__builtins__': None}
                           )
        if index is None:
            self.inputs.append(element)

        elif index > len(self.inputs) - 1:
            print("Index ({}) out of range, appending {} to the end."
                  .format(index, element))
            self.inputs.append(element)

        else:
            self.inputs.insert(index, element)

    def delete(self, element=None):  # [check]
        """
        Deletes the element given and if the element is
        not given, then remove the last value
        """
        if element is not None:
            return self.inputs.remove(element)

        else:
            return self.inputs.pop()

    def write(self):  # [check]
        with open(self.fn, "r+") as f:
            f.write("header\n")
            f.write("time to stop\n")

            for inpt in self.inputs:
                if isinstance(inpt, (int, float)):
                    inpt = "{:.4f}".format(inpt)

                elif isinstance(inpt, (list, np.ndarray)):
                    empty = ""

                    for inp in inpt:
                        empty = empty + "{:.4f} ".format(inp)

                    inpt = empty

                elif isinstance(inpt, str):
                    inpt = inpt.strip("\n").join("''")

                if inpt[-1] is not "\n":
                    inpt = inpt + "\n"

                f.writelines(inpt)

            f.write("\\E")

