from parameters import parameters
import numpy as np


class parameterFile():

    def __init__(self, name, fname, priority, params):
        self.name = name
        self.fn = fname
        self.priority = priority
        self.params = params
        self.size = len(params)

    def resetStates(self):
        for param in self.params:
            param.changeState(False)

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

        parameterList = (parameters.atomic_weight(ZERO),
            parameters.charge(ZERO))

        return parameterList

    @staticmethod
    def emptyBackground():
        ZERO = np.array([0], dtype=np.float)

        parameterList = (parameters.ne_numTimePts(ZERO),
                    parameters.ne_timePts(ZERO),
                    parameters.ne_paramType(ZERO),
                    parameters.ne_radCoord(ZERO),
                    parameters.ne_numInterpPts(ZERO),
                    parameters.ne_radGrid(ZERO),
                    parameters.ne_radGridPts(ZERO),
                    parameters.ne_decayLength(ZERO),
                    parameters.te_numTimePts(ZERO),
                    parameters.te_timePts(ZERO),
                    parameters.te_paramType(ZERO),
                    parameters.te_radCoord(ZERO),
                    parameters.te_numInterpPts(ZERO),
                    parameters.te_radGrid(ZERO),
                    parameters.te_radGridPts(ZERO),
                    parameters.te_decayLength(ZERO),
                    parameters.ti_numTimePts(ZERO),
                    parameters.ti_timePts(ZERO),
                    parameters.ti_paramType(ZERO),
                    parameters.ti_radCoord(ZERO),
                    parameters.ti_numInterpPts(ZERO),
                    parameters.ti_radGrid(ZERO),
                    parameters.ti_radGridPts(ZERO),
                    parameters.ti_decayLength(ZERO))

        return parameterList
