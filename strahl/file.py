from parameters import parameters

import auxillary as aux
import subprocess as sub
import numpy as np

import os
import sys
import re

# TODO
# COMPARTEMENTALIZE METHODS INTO A 'file' class


class parameterFile():

    def __init__(self, name, fname, priority, params):
        self.type = name
        self.fn = fname
        self.priority = priority
        self.paramDict = params
        self.size = len(params)

    def attributes(self, all=False):
        print("Attributes of {} object {}:".format(self.__class__, self))

        print(self.__dict__)

        if all:
            [param.attributes() for param in self.paramDict.values()]

    def resetStates(self):
        for param in self.paramDict.values():
            param.changeState(False)

    def attribute_dictionary(self, keys=None, param_keys=None, dic=None):
        attributes = dict()

        if dic is None or not isinstance(dic, dict):
            dic = self.__dict__

        if keys is None or not isinstance(keys, list):
            keys = self.__dict__.keys()

        for key in keys:
            val = dic[key]

            if isinstance(val, dict):
                keys = list(val.keys())
                val = self.attribute_dictionary(keys, param_keys, val)

            if isinstance(val, parameters):
                val = val.attribute_dictionary(param_keys)

            if isinstance(val, np.ndarray):
                val = val.tolist()

            attributes[key] = val

        return attributes

    @classmethod
    def mainFile(cls, fname):
        params = parameterFile.emptyMain()

        return cls("main", fname, 1, params)

    @classmethod
    def backgroundFile(cls, fname):
        params = parameterFile.emptyBackground()

        return cls("background", fname, 2, params)

    @classmethod
    def geometryFile(cls, fname):
        return cls("geometry", fname, 3, list())

    @classmethod
    def fluxFile(cls, fname):
        return cls("flux", fname, 4, list())

    @staticmethod
    def emptyMain():
        ZERO = np.array([0], dtype=np.float)

        parameterList = {"atomic weight": parameters.atomic_weight(ZERO),
                        "charge": parameters.charge(ZERO),
                        "shot": parameters.shot(ZERO),
                        "index": parameters.index(ZERO)}

        return parameterList

    @staticmethod
    def emptyBackground():
        ZERO = np.array([0], dtype=np.float)

        parameterList = {"time-vector": parameters.ne_numTimePts(ZERO),
                        "time-vector": parameters.ne_timePts(ZERO),
                        "time-vector": parameters.te_numTimePts(ZERO),
                        "time-vector": parameters.te_timePts(ZERO),
                        "time-vector": parameters.ti_numTimePts(ZERO),
                        "time-vector": parameters.ti_timePts(ZERO)}

        # parameterList = (parameters.ne_numTimePts(ZERO),
        #             parameters.ne_timePts(ZERO),
        #             parameters.ne_paramType(ZERO),
        #             parameters.ne_radCoord(ZERO),
        #             parameters.ne_numInterpPts(ZERO),
        #             parameters.ne_radGrid(ZERO),
        #             parameters.ne_radGridPts(ZERO),
        #             parameters.ne_decayLength(ZERO),
        #             parameters.te_numTimePts(ZERO),
        #             parameters.te_timePts(ZERO),
        #             parameters.te_paramType(ZERO),
        #             parameters.te_radCoord(ZERO),
        #             parameters.te_numInterpPts(ZERO),
        #             parameters.te_radGrid(ZERO),
        #             parameters.te_radGridPts(ZERO),
        #             parameters.te_decayLength(ZERO),
        #             parameters.ti_numTimePts(ZERO),
        #             parameters.ti_timePts(ZERO),
        #             parameters.ti_paramType(ZERO),
        #             parameters.ti_radCoord(ZERO),
        #             parameters.ti_numInterpPts(ZERO),
        #             parameters.ti_radGrid(ZERO),
        #             parameters.ti_radGridPts(ZERO),
        #             parameters.ti_decayLength(ZERO))

        return parameterList


class inputFile():  # [check]

    def __init__(self, inpt_fn, inputs=None):
        self.create(inpt_fn)

        if inputs is not None and isinstance(inputs, list):
            self.inputs = inputs
        else:
            self.inputs = list()

    def create(self, inpt_fn=None):
        new_inpt_prompt = "Please enter a new input file or exit [enter]? "

        if inpt_fn is None:
            inpt_fn = input(new_inpt_prompt)

        if os.path.isfile(inpt_fn):
            print("Error: The file {} already exists."
                  .format(aux.colorFile(inpt_fn)))

            overwrite = input("Would you like to overwrite {} [Y/N]: "
                              .format(aux.colorFile(inpt_fn)))

            if overwrite is "":
                sys.exit("Exiting")

            if re.match("(n|N)", overwrite):
                aux.print_dirContents(os.curdir)

                inpt_fn = input(new_inpt_prompt)

                if inpt_fn is "":
                    sys.exit("Exiting")

                else:
                    return self.create(inpt_fn)

            elif re.match("(y|Y)", overwrite):
                with open(inpt_fn, "w") as f:
                    f.truncate(0)

        mkfileCmd = "touch {}".format(inpt_fn)
        sub.call(mkfileCmd.split())

        self.fn = inpt_fn
        self.path = os.path.join(os.curdir, inpt_fn)

    def attributes(self, all=False):
        print("Attributes of {} object {}:".format(self.__class__, self))

        print(self.__dict__)


class summaryFile():

    def __init__(self, sum_fn=None):
        self.create(sum_fn)

    def create(self, sum_fn=None):
        if sum_fn is None:
            aux.print_dirContents("./summaries")

            sum_fn = input("Please select or create a summary file: ")

        serial = re.search("\\.(json|hdf5)", sum_fn)

        if serial is None:
            print("{}".format(sum_fn))

            print("Your file name must end in .json or .hdf5.")

            sum_fn = input("Enter a new file name or exit [enter]: ")

            if serial is "":
                sys.exit("Exiting")

            else:
                return self.create(sum_fn)

        path_to_file = os.path.join(os.curdir, "summaries", sum_fn)

        if not os.path.isfile(path_to_file):

            mkFileCmd = "touch {}".format(path_to_file)

            sub.call(mkFileCmd.split())

        self.path = os.path.dirname(path_to_file)
        self.fn = sum_fn
        self.serial = serial.group(1)

    def attributes(self):
        print("Attributes of {}".format(self))

        print(self.__dict__)



