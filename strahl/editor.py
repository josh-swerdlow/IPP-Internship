from parameterFile import parameterFile
import subprocess as sub
import os
import re
import sys
import pickle as rick
import auxillary as aux
import shutil as shell


class FileEditor():

    def __init__(self, file):
        self.fileName = file
        self._deletions = 0
        self._line = 0

        if not os.path.isdir("./_backup"):
            mkdirBckupCmd = "mkdir ./_backup"
            sub.call(mkdirBckupCmd.split())

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
            sub.call(delLineCmd.split())
            self.update()
        elif deletion is not empty:  # delete the deletion
            if line is 0:
                line = self._line + 1
            delPhraseCmd = ("sed -l -i.bck_up{} {}s/{}/{}/{} {}"
                .format(self._deletions, line, deletion,
                        replacement, repeat, self.fileName))
            print(delPhraseCmd + "\n")
            sub.run(delPhraseCmd.split())
            self.update()
        else:
            print("Broke af")

        if backup:
            mvBckupCmd = ("mv {}.bck_up{} ./_backup"
                .format(self.fileName, self._deletions))
            sub.call(mvBckupCmd.split())
        else:
            delBckupCmdn = (("rm {}.bck_up{}")
                .format(self.fileName, self._deletions))
            sub.call(delBckupCmdn.split())

    def closeFile(self, clean=1):
        """
        Closes the file and by default deletes the
        backups for this file in the folder ./_backup
        """
        if clean:
            cleanBckupDir = "rm ./_backup/{}.bck_up*".format(self.fileName)
            sub.call(cleanBckupDir.split())
        self.file.close()


class paramFileEditor(FileEditor):
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


class inputFileEditor(FileEditor):
    """
    TODO: Port to pysqllite? json?
    """

    def __init__(self, inpt_fn, sum_fn, parameterFiles):
        # if fn DNE, create fn + sum
        # if not os.path.isfile(fn):

        #     print("Error: {} was not found in ~/{}\n" +
        #         " This is the current directory.\n"
        #         .format(aux.colorFile(fn), aux.colorDir(os.curdir)))

        #     aux.print_dirContents(os.curdir)

        #     fn = input("Please enter a different input file, " +
        #             "create a new input file, " +
        #             "or exit [enter]: ")

        #     if fn is '':
        #         print("Exiting program")
        #         sys.exit()
        #     else:
        #         if not os.path.isfile(fn):
        #             mkfile = "touch {}".format(fn)
        #             sub.call(mkfile.split())
        #         self.__init__(fn)

        self.inpt_fn = inpt_fn
        self.sum_fn = sum_fn
        self.parameterFiles = parameterFiles

    def create_inptFile(inpt_fn=None):

        if inpt_fn is None:
            inpt_fn = input("What would you like to call your input file? ")

        mkfile = "touch {}".format(inpt_fn)
        sub.call(mkfile.split())

        return inpt_fn

    def create_sumFile(sum_fn=None):
        if sum_fn is None:
            sum_fn = input("Your summary file will be given " +
                    "a file extension and be put into the directory {}".
                    format(aux.colorDir("./summaries")) +
                    "What would you like to call your summary file? ")

        sum_fn = sum_fn + ".pckl"
        mkfile = "touch ./summaries/{}".format(sum_fn)
        sub.call(mkfile.split())

        return sum_fn

    def load_sumFile(sum_fn=None):
        """
        If the file has a .pckl then, just entering the name will work
        unless there exist a file with the same name. So if you intend
        to give the load function a file without a .pckl ending, just
        type that file name and it will still work.
        """
        sum_dir = os.path.join(os.curdir, "summaries")

        if sum_fn is not None:
            if not re.match(".*.pckl", sum_fn):
                sum_fn = sum_fn + ".pckl"

            sum_path = os.path.join(sum_dir, sum_fn)

        if sum_fn is None or not os.path.isfile(sum_path):

                print("Error: {} was not found in {}\n"
                    .format(aux.colorFile(sum_fn),
                        aux.colorDir(sum_dir)) +
                    "This is the {} directory.\n"
                    .format(aux.colorDir(sum_dir)))

                aux.print_dirContents(sum_dir)

                newSum_fn = input("Please enter a different summary file, " +
                        "or exit [enter]: ")
                newSum_path = os.path.join(sum_dir, newSum_fn)

                print("New Summary file is {}:".format(newSum_fn))
                print("New Summary path is {}:".format(newSum_path))

                # TODO: Add a literal option
                if newSum_fn is "":  # [check]
                    print("Exiting program")
                    sys.exit()

                if not os.path.isfile(newSum_path):
                    newSum_fn = inputFileEditor.load_sumFile(newSum_fn)

                sum_fn = newSum_fn

                # print("Using {} as summary file".
                # format(aux.colorPckl(sum_fn)))

        return sum_fn

    @classmethod
    def load(cls, inpt_fn=None, *parameterFiles):
        if inpt_fn is not None and os.path.isfile(inpt_fn):
            print("{} is an input file found in {}"
                .format(aux.colorFile(inpt_fn), aux.colorDir(os.curdir)))

            new = input("Would you like to overwrite {}, "
                        .format(aux.colorFile(inpt_fn)) +
                        "load it into a new file, or exit [Y/N/Enter]? ")

            if re.match("(y|Y)", new):  # [check]
                sum_fn = cls.load_sumFile(inpt_fn)

            elif re.match("(n|N)", new):  # [todo]
                sum_fn = os.path.join("summaries",
                            inputFileEditor.load_sumFile(inpt_fn))

                newInpt_fn = inputFileEditor.create_inptFile()
                newSum_fn = os.path.join("summaries",
                            inputFileEditor.create_sumFile(inpt_fn))

                shell.copyfile(inpt_fn, newInpt_fn)
                shell.copyfile(sum_fn, newSum_fn)

            elif new is "":  # [check]
                print("Exiting Program")
                sys.exit()

            else:  # [check]
                print("Error: incorrect answer given...trying again.\n\n")
                cls.load(inpt_fn)

        else:  # [check]
            print("Error: {} was not found in {}\n"
                .format(aux.colorFile(inpt_fn),
                    aux.colorDir("~/" + os.curdir)) +
                "This is the current directory.\n")

            aux.print_dirContents(os.curdir)

            inpt_fn = input("Please enter a different input file, " +
                    "create a new input file [new file name], " +
                    "or exit [enter]: ")

            if inpt_fn is '':  # [check]
                print("Exiting program")
                sys.exit()

            else:
                if not os.path.isfile(inpt_fn):  # [check]
                    inpt_fn = inputFileEditor.create_inptFile(inpt_fn)
                    sum_fn = inputFileEditor.create_sumFile(inpt_fn)
                    print("Creating {} and {} as input and summary files"
                        .format(aux.colorFile(inpt_fn),
                            aux.colorPckl(sum_fn)))

                else:  # [check]
                    cls.load(inpt_fn)

        return cls(inpt_fn, sum_fn)

    def create_summary(self, summary_file):
        return open(summary_file, "wb")  # Open with write binary

    def close_summary(self, summary):
        summary.close()

    def load_summary(self, sum_fn):
        """
        This is self contained as does not require create_summary or
        close_summary methods.
        In order to get the previous information.
        returns the unpickled object
        """
        load_obj = None

        with open(sum_fn, "rb") as summary:  # Open with read binary
            load_obj = rick.load(summary)

        return load_obj

    def summarize(self, obj, summary_file=None):
        """
        Summarizes the object into the summary_file.
        If this is not given, then uses self.sum_fn
        """


    def update_summary(self, new_obj, summary_file=None, summary=None):
        """
        Can either write new_obj to a summary obj which is already open
        or to a summary_file to overwrite it.
        """
        if summary_file is None and summary is None:
            print("Either summary_file and summary must be passed.")
            raise TypeError
            return -1

        if summary_file is not None:
            with open(summary_file, "wb") as summary:
                rick.dump(new_obj, summary, rick.HIGHEST_PROTOCOL)

        elif summary is not None:
            rick.dump(new_obj, summary, rick.HIGHEST_PROTOCOL)

    @staticmethod
    def basic_write(self):
        pass

    @staticmethod
    def blankLines(self):
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



