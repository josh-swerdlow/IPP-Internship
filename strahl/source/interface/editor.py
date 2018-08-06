########################################
# File name: parameters.py             #
# Author: Joshua Swerdow               #
# Date created: 5/20/2018              #
# Date last modified:                  #
# Python Version: 3.0+                 #
########################################

"""
Summary: Stores the file, parameter file, input file, and
    summary file editor classes. These generally order
    information and functions that create, interact, or
    load the parameter, input, or summary objects.

To Do:
    Comment code!
    fileEditor - [x]
    ParameterFileEditor - [x]
    SummaryFileEditor - [x]
    InputFileEditor - []
"""

__author__ = 'Joshua Swerdow'

import os
import re
import sys
import json


import numpy as np
import subprocess as sub
import source.interface.query as query

from source.interface.file import ParameterFile, InputFile, SummaryFile


class FileEditor():
    """
    Summary: This object wraps around the standard Python file object
    to add further functionality specific for reading and writing
    to parameter files.

    To Do:
        Figure out back up files?
        No verbose output at the moment
    """

    def __init__(self, fn):
        """
        Initializes the FileEditor object.

        Parameters:
            * **file** [str]: the name of the file to be opened.

        Attributes:
            * **fn** [str]: the name of the file opened.
            * **_deletions** [int|0]: the current number of
                deletions performed for use in backup file.
            * **_line** [int|0]: the current line number.
        """
        self.fn = fn
        self._deletions = 0
        self._line = 0

        # if not os.path.isdir("./_backup"):
        #     mkdirBckupCmd = "mkdir ./_backup"
        #     sub.call(mkdirBckupCmd.split())

    def open_file(self, read_write_attribute="r+", buffer_time=1):
        """
        Opens the given file and creates a file standard Python
        file object.

        Parameters:
            * **read_write_attribute** [str|"r+"]: sets the
                read and write abilities of the file object.
            * **buffer_time**[int|1]: sets the buffer time.

        Attributes:
            * **file** [file obj]: a file object
        """

        self.file = open(self.fn, read_write_attribute)

    def read_char(self, n=1):
        """
        Returns n characters read from the buffer and keeps track
        of the current line.

        Parameters:
            * **n** [int|1]: number of characters to be read

        Raises:
            OSError: If the file object is not readable

        Note: This should be used to capture select words from a
            setence, rather than read line or lines. For that use
            FileEditor.readline().
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

    def read_lines(self, n=1):
        """
        Returns the line read from the buffer and keeps track of
        the current line.

        Parameters:
            * **n** [int|1]: The number of lines to read

        Raises:
            OSError: If the file object is not readable.
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

    def cur_loc(self):
        """
        Returns the current location in the form of the current
        character and the current line.

        Raises:
            OSError: If the file object is not seekable
        """
        if self.file.seekable():
            return (self.file.tell(), self._line)

        else:
            print("File cannot be randomly accessed.")
            raise OSError

    def move(self, loc=(0, 0)):
        """
        Resets the buffer to the specified location relative to the
        begining of the buffer and resets the line attribute to the
        line at the reset location.

        Parameters:
            * **loc** [tuple of int|(0,0)]: the location to move to

        Raises:
            OSError: If the file object is not seekable
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
        Reads n characters read from the buffer, but returns
        to the initial location.

        Parameters:
            * **n** [int|1]: the number of characters to read

        Returns:
            chars: The characters read from the file.
        """
        loc = self.cur_loc()
        chars = self.read_char(n)

        self.move(loc)

        return chars

    def peak_lines(self, n=1):
        """
        Reads until a newline is found or EOF, but returns to the
        initial location.

        Parameters:
            * **n** [int|1]: the number of lines to read

        Returns:
            line: The lines read from the file.
        """
        loc = self.cur_loc()
        line = self.read_lines(n)

        self.move(loc)

        return line

    def update(self):
        """Updates the buffer immediately"""
        self.file.flush()

    def add(self, addition):
        """
        Adds the given addition to the file at the current location.

        Parameters:
            * **addition** [str]: The addition to be added to the file.

        Raises:
            OSError: If the file is not writable.
        """
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
        Deletes the given deletion phrase or by default a single line.
        The previous file is always saved into a folder called ./_backup.

        Paramaters:
            * **deletion** [str|""]: The string to be deleted.
            * **replacement** [str|""]: The string to be replaced
            * **line** [int|0]: The number of lines to be deleted
            * **repeat** [int|1]: The numbe of repeations
        """
        empty = ""
        self._deletions += 1

        if deletion is empty:
            delLineCmd = ("sed -l -i.bck_up{} {}d {}"
                .format(self._deletions, self._line + 1, self.fn))

            sub.call(delLineCmd.split())

            loc = self.cur_loc()

        elif deletion is not empty:  # delete the deletion
            if line is 0:
                line = self._line + 1

            delPhraseCmd = ("sed -l -i.bck_up{} {}s/{}/{}/{} {}"
                .format(self._deletions, line, deletion,
                        replacement, repeat, self.fn))

            print(delPhraseCmd + "\n")

            sub.run(delPhraseCmd.split())

            loc = self.cur_loc()

        else:
            print("Broke af")

        self.file.close()
        self.open_file()
        self.move(loc)

        # if backup:
        #     mvBckupCmd = ("mv {}.bck_up{} ./_backup"
        #         .format(self.fn, self._deletions))
        #     sub.call(mvBckupCmd.split())
        # else:
        #     delBckupCmdn = (("rm {}.bck_up{}")
        #         .format(self.fn, self._deletions))
        #     sub.call(delBckupCmdn.split())

    def close_file(self, clean=True):
        """
        Closes the file and by default deletes the
        backups for this file in the folder ./_backup
        """
        if clean:
            cleanBckupDir = "rm ./_backup/{}.bck_up*".format(self.fn)
            sub.call(cleanBckupDir.split())
        self.file.close()


class ParameterFileEditor(FileEditor, ParameterFile):
    """
    Summary: A class that handles general editing of parameter files.
    It is built on top of the FileEditor() class.

    To Do:
        Needs to deal with writing and deleting comments
        What exactly is next doing? haha
    """

    def __init__(self, file_path, parameter_file, verbosity=False):
        """
        Initializes a ParameterFileEditor object

        Parameters:
            * **param_fn** [str]: The parameter file name.
            * **verbosity** [bool|False]: turns on and off verbose
                ouput.

        Attributes:
            * **verbose** [bool|False]: determines if verbose
                execution is used.
        """
        super().__init__(file_path)

        self.parameter_file = parameter_file

        self.open_file()

        self.verbose = verbosity
        if self.verbose:
            print(">>>>>Initialized a parameter file editor for {}"
                 .format(file_path))

    @classmethod
    def create_editors(cls, *file_names,
                       verbosity=False):
        """
        Loads multiple ParameterFileEditor objects depending on the file_names.
        Only accepts one file_name of each file_type and only accepts
        file_type in the following order: main, background, geometry, flux.

        Parameters:
            * **file_names** [list]: list of parameter file names
                expects one of each (main, background, geometry, and flux)
            * **verbosity** [bool|False]: turns on and off verbose ouput

        Returns:
            A list of ParameterFileEditor objects for the file names
            given in order of main, bckg, geom, flux. Returns an
            empty list if no file names are given.
        """
        if len(file_names) == 0:
            sys.exit("No file names given to create paramater editors.")

        if not isinstance(file_names, list):
            file_names = list(file_names)

        main_dir = "param_files"
        bckg_dir = "nete"
        geom_dir = bckg_dir
        flux_dir = bckg_dir

        directories = [main_dir, bckg_dir, geom_dir, flux_dir]

        main = None
        bckg = None
        geom = None
        flux = None

        parameter_editors = [main, bckg, geom, flux]

        for index in range(len(file_names)):
            if file_names[index] is not "":

                if os.path.dirname(file_names[index]) is not '':
                    file_names[index] = os.path.basename(file_names[index])

                path = os.path.join(directories[index], file_names[index])

                if os.path.isfile(path):

                    editor = cls.main_editor(file_names[index],
                                           path,
                                           verbosity)

                    parameter_editors[index] = editor

        # if bckg_fn is not "":
        #     bckg_path = os.path.join(bckg_dir, bckg_fn)
        #     if os.path.isfile(bckg_path):

        #         bckg = cls.bckg_editor(bckg_fn,
        #                                bckg_path,
        #                                verbosity)

        #         parameter_editors[1] = bckg

        # if geom_fn is not "":
        #     geom_path = os.path.join(geom_dir, geom_fn)
        #     if os.path.isfile(geom_path):

        #         geom = cls.geom_editor(geom_fn,
        #                                geom_path,
        #                                verbosity)

        #         parameter_editors[2] = geom

        # if flux_fn is not "":
        #     flux_path = os.path.join(flux_dir, flux_fn)
        #     if os.path.isfile(flux_path):

        #         flux = cls.flux_editor(flux_fn,
        #                                flux_path,
        #                                verbosity)

        #         parameter_editors[3] = flux

        return parameter_editors

    @classmethod
    def main_editor(cls, fn, file_path, verbosity):
        parameter_file = ParameterFile.create_parameter_file(fn,
                                                             file_path,
                                                             "main",
                                                             verbosity)

        return cls(file_path, parameter_file, verbosity)

    @classmethod
    def bckg_editor(cls, fn, file_path, verbosity):
        parameter_file = ParameterFile.create_parameter_file(fn,
                                                             file_path,
                                                             "background",
                                                             verbosity)

        return cls(file_path, parameter_file, verbosity)

    @classmethod
    def geom_editor(cls, fn, file_path, verbosity):
        parameter_file = ParameterFile.create_parameter_file(fn,
                                                             file_path,
                                                             "geometry",
                                                             verbosity)

        return cls(file_path, parameter_file, verbosity)

    @classmethod
    def flux_editor(cls, fn, file_path, verbosity):
        parameter_file = ParameterFile.create_parameter_file(fn,
                                                             file_path,
                                                             "flux",
                                                             verbosity)

        return cls(file_path, parameter_file, verbosity)

    def search_parameter_file(self, search_key="comments"):
        """
        Searches the entire parameter file to turn on and off the
        parameters which have been elected to be changed with comments ('#').

        Parameters:
            * **paramFile** [ParameterFile obj]: a ParameterFile object
                to search through
            * **search_key** [str|"comments"]: chooses to search
                for comments or non-comments
            * **verbose** [bool|False]: turn on commented output
        """
        # Search entire file for comments and non comments
        # then change the relevant parameters
        keywords = set()
        while keywords is not None:
            if search_key == "noncomments":
                if self.verbose:
                    print("Searching parameter file {} for non-comments"
                        .format(self.parameter_file.fn))

                keywords = self._search_parameter_file_non_comments()

            else:
                if self.verbose:
                    print("Searching parameter file {} for comments"
                        .format(self.parameter_file.fn))

                keywords = self._search_parameter_file_comments()

            # Change the relevant parameters if there are any
            if keywords is not None and len(keywords) != 0:
                self._change_relevant_parameters(self.parameter_file, keywords)

    def _next(self, regex):
        """
        Moves to the first next regex in the given file and
        returns a boolean value if found.

        Paramaters:
            * **regex** [str]: A regular expression phrase to search for.

        Returns:
            True: If the regex is found otherwise false
            None: If we are at the end of the file already
        """
        while self.peak_lines() != '':
            line = self.peak_lines()

            if re.match(regex, line) is not None:
                return True
                break

            else:
                self.read_lines()
                return False

        return None

    def _extract_parameters(self, line, regex):
        """
        Given a parameter header input, this extracts the parameters
        according to the regex given.

        Parameters:
            * **line** [str]: the parameter header input line of text
            * **regex** [str]: a regular expression search string

        Returns:
            A set of parameters from the input line or -1 if the
            parameter header was not handled correctly.
        """
        split_temp = re.split(regex, line)

        temp = ""
        if len(split_temp) == 2:
            temp = split_temp[1][0:-1]  # Remove the newline character

            if self.verbose:
                print("Splitting the parameter header ...")

        else:
            print("The parameter header was not split correclty.")

            if self.verbose:
                print(split_temp)

            return -1

        paramsSet = set(re.split("\\s{2,}", temp))

        if self.verbose:
            print("Returning a set of parameters from the header input line: ")
            print(paramsSet)

        return paramsSet

    def _next_noncomment(self):
        """
        Moves to the next 'cv ' in the given file.

        Returns:
            A call to self.next with a regex phrase for noncomments.
        """
        if self.verbose:
                print("Moving to next noncomment ...")

        return self._next("(cv\\s){1}")

    def _next_comment(self):
        """
        Moves to the next 'cv#' in the given file.

        Returns:
            A call to self.next with a regex phrase for comments.
        """
        if self.verbose:
            print("Moving to next comment ...")

        return self._next("(cv#)")

    def _extract_non_commented_parameters(self, line):
        """
        Given a parameter header input, this extracts the
        uncommented parameters

        Returns:
            A call to self.extractParameters with a regex phrase
            for noncomments.
        """
        if self.verbose:
            print("Extracting uncommented parameters ...")

        return self._extract_parameters(line, "cv\\s*")

    def _extract_commented_parameters(self, line):
        """
        Given a parameter header input, this extracts the
        commented parameters

        Returns:
            A call to self.extractParameters with a regex phrase
            for noncomments.
        """
        if self.verbose:
            print("Extracting commented parameters ...")

        return self._extract_parameters(line, "cv#\\s*")

    def _search_parameter_file_comments(self):
        """
        Given an ParameterFileEditor object go to the next comment
        line, extract the commented parameters from that line, and
        return them in a set of keywords.

        Returns:
            * **keywords** [set]: A set of keywords found in the
                parameter file
        """
        keywords = set()

        next_comment = self._next_comment()

        if next_comment is None:
            keywords = None

        elif next_comment:
            line = self.read_lines()

            keywords = self._extract_commented_parameters(line)

        return keywords

    def _search_parameter_file_non_comments(self):
        """
        Given an ParameterFileEditor object go to the next non-comment
        line, extract the non-commented parameters from that line, and
        return them in a set of keywords.

        Returns:
            * **keywords** [set]: A set of keywords found in the
                parameter file
        """
        keywords = set()

        next_comment = self.next_noncomment()

        if next_comment is None:
            keywords = None

        elif next_comment:
            line = self.read_lines()

            keywords = self.extract_non_commented_parameters(line)

        return keywords

    def _change_relevant_parameters(self, parameter_file, keywords):
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
        parameters = set(parameter_file.parameter_dict.keys())
        relev_parameters = parameters & keywords

        # Change parameters to correct state
        [parameter_file.parameter_dict[param_name].change_state()
            for param_name in relev_parameters]


class SummaryFileEditor(SummaryFile):
    """
    Summary: A class that handles 'get'-ing and 'write'-ing
    of summary files. It is built on top of the SummaryFile() class.

    To Do:
        Add HDF5
    """

    def __init__(self, sum_fn, verbosity=False):
        """
        Initializes a summary file editor object.

        Parameters:
            * **sum_fn** [str]: The summary files name
            * **verbosity** [bool|False]: turns on and off verbose ouput

        Attributes:
            * **loadable** [bool|False]: determines if something
                has been written to this summary file.
            * **verbose** [bool|False]: determines if verbose
                execution is used.
        """
        super().__init__(sum_fn, verbosity=verbosity)

        self.loadable = False

        self.verbose = verbosity
        if self.verbose:
            print(">>>>>Initialized a summary file editor object for {}"
                 .format(sum_fn))

    def get(self):
        """
        Gets the summary file using the serialization on the file
        extension.

        Returns:
            The Python converted JSON dictionary that was
            read in from the summary file. If the object cannot
            be converted then None is returned.
        """
        obj = None

        path_to_file = os.path.join(self.path, self.fn)

        if self.serial == "json":
            if self.verbose:
                print("Getting with json serialization ...")

            obj = self._getJSON(path_to_file)

        elif self.serial == "hdf5":
            if self.verbose:
                print("Getting with hdf5 serialization ...")

            obj = self._getHDF5(path_to_file)

        else:
            print("Incorrect serial value: {}".format(self.serial))

        if self.verbose:
            print("Got {} from {}".format(obj, path_to_file))

        return obj

    def write(self, dic):
        """
        Writes the input file into a summary file using the
        serialization on the file extension.

        Parameters:
            * **dic** [dic]: A dictionary summary of the object
                which is created using custom method for the
                file and parameter objects.

        """
        path_to_file = os.path.join(self.path, self.fn)

        if self.serial == "json":
            if self.verbose:
                print("Writing with json serialization ...")

            self._writeJSON(dic, path_to_file)

        elif self.serial == "hdf5":
            if self.verbose:
                print("Writing with hdf5 serialization ...")

            self._writeHDF5(dic, path_to_file)

        else:
            print("Incorrect serial value: {}".format(self.serial))

        if self.verbose:
            print(">>>>>Created the following summmary file in {}".format(path_to_file))
            with open(path_to_file, "r") as f:
                print(f.read(-1))

        self.loadable = True

    @staticmethod
    def _getJSON(fn):
        """
        Gets the obj dictionary from the json file given.

        Parameters:
            * **fn** [str]: a json files name

        Returns:
            A Python dictionary saved in the JSON file.
        """
        obj = None

        with open(fn, "r") as f:
            obj = json.load(f)

        return obj

    @staticmethod
    def _writeJSON(dic, fn):
        """
        Write the given dictionary to the json file given.

        Parameters:
            * **dic** [dic]: A python dictionary describing some class.
            * **fn** [str]: The file name to write into.
        """

        with open(fn, "w") as f:
            json.dump(dic, f, indent=4)

    @staticmethod
    def _getHDF5(obj):
        pass

    @staticmethod
    def _writeHDF5(obj):
        pass


class InputFileEditor(InputFile):

    def __init__(self, inpt_fn=None, inputs=None, verbosity=False):
        """
        Initializes an InputFileEditor object

        Parameters:
            * **inpt_fn** [str|None]: The input file name.
            * **inputs** [list|None]: the list of values of the parameters
            * **verbosity** [bool|False]: turns on and off verbose
                ouput.

        Attributes:
            * **verbose** [bool|False]: determines if verbose
                execution is used.
        """
        super().__init__(inpt_fn, inputs, verbosity=verbosity)

        self.verbose = verbosity
        if self.verbose:
            print(">>>>>Initialized an input file editor object.")

    def add(self, element=None, index=None):
        """
        Adds a value (if given) to the input file. If it is not given,
        then the user is asked to input a value which is used.

        Parameters:
            * **element** [???|None]: The element that should be
                added to the input file.
            * **index** [int|None]: The location in the input file
                that the element should be placed.
        """
        if element is None:
            element = query.evaluate()

        if index is None:
            if self.verbose:
                print("Appending {} to the end of the input file values."
                     .format(element))

            self.inputs.append(element)

        elif index > len(self.inputs) - 1:
            print("Index ({}) out of range, appending {} to the end."
                  .format(index, element))

            self.inputs.append(element)

        else:
            if self.verbose:
                print("Add {} into index ({})"
                     .format(element, index))

            self.inputs.insert(index, element)

    def delete(self, element=None):
        """
        Deletes the element given and if the element is
        not given, then remove the last value.

        Parameters:
            * **element** [???|None]: The element to be removed

        Returns:
            The output of either removing the given element or
            simply popping from the inputs list.
        """
        if element is not None:
            if element in self.inputs:
                return self.inputs.remove(element)

            else:
                print("{} is not in inputs"
                     .format(element))
        else:
            if self.verbose:
                print("Removing the last value in inputs")

            return self.inputs.pop()

    def write(self, main_fn):
        """
        Writes the inputs list to a text file given by the file name.
        During this process we check every possible instance of date
        type in the inputs list and convert them to strings which can
        be read by STRAHL natively.
        """
        if self.verbose:
            print("Writing the input file ...")

        with open(self.fn, "r+") as f:
            f.write(main_fn + "\n")
            f.write("0.00\n")

            for inpt in self.inputs:
                inpt_bckup = inpt

                if isinstance(inpt, (int, float)):
                    inpt = "{:.4f}".format(inpt)

                    if self.verbose:
                        print("{} was an instance of int or float"
                             .format(inpt_bckup))
                        print("Changing {} to {}"
                             .format(inpt_bckup, inpt))

                elif isinstance(inpt, (list, tuple)):
                    empty = ""

                    for inp in inpt:
                        empty = empty + "{:.4f} ".format(inp)

                    inpt = empty

                    if self.verbose:
                        print("{} was an instance of list or tuple"
                             .format(inpt_bckup))
                        print("Changing {} to {}"
                             .format(inpt_bckup, inpt))

                # Currently only handles 2D arrays
                elif isinstance(inpt, np.ndarray):
                    empty = ""

                    if inpt.ndim > 2:
                        print("Error: {} has more than 2 dimensions"
                            .format(inpt))

                    # Pad inpt until it has two dimensions so one can
                    # treat it like a 2D array regardless of the values

                    shape = list(inpt.shape)
                    while inpt.ndim < 2:
                        shape.append(1)
                        inpt = np.expand_dims(inpt, 0)
                        print(inpt)

                    cols = shape[0]
                    rows = shape[1]

                    empty = ""
                    for row in range(rows):
                        row_str = str()
                        for col in range(cols):
                            row_str = row_str + "{:.4f} ".format(inpt[row, col])

                        row_str = row_str[:-1] + "\n"

                        empty = empty + row_str

                    inpt = empty

                    if self.verbose:
                        print("{} was an instance of np.ndarray"
                              .format(inpt_bckup))
                        print("Changing {} to {}".format(inpt_bckup, inpt))

                elif isinstance(inpt, str):
                    inpt = inpt.strip("\n").join("''")

                    if self.verbose:
                        print("{} was an instance of a string"
                             .format(inpt_bckup))
                        print("Changing {} to {}"
                             .format(inpt_bckup, inpt))

                if not inpt.endswith("\n"):
                    if self.verbose:
                        print("Adding a newline character")

                    inpt = inpt + "\n"

                f.writelines(inpt)

            f.write("\\E")

            if self.verbose:
                print(">>>>>Created the following input file")
                f.seek(0)
                print(f.read(-1))

