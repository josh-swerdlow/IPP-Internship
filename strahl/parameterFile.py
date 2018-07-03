from inputParameter import inputParameter
import numpy as np


class parameterFile():

    def __init__(self, name, priority, params):
        self.name = name
        self.priority = priority
        self.params = params
        self.size = len(params)

    @classmethod
    def mainFile(cls):
        params = parameterFile.emptyMain()

        return cls("mainFile", 1, params)

    @classmethod
    def backgroundFile(cls):
        params = parameterFile.emptyBackground()

        return cls("backgroundFile", 2, params)

    @classmethod
    def geometryFile(cls):
        return cls("geometryFile", 3, list())

    @classmethod
    def fluxFile(cls):
        return cls("fluxFile", 4, list())

    @staticmethod
    def emptyMain():
        ZERO = np.array([0], dtype=np.float)

        parameters = (inputParameter.atomic_weight(ZERO),
            inputParameter.charge(ZERO))

        return parameters

    @staticmethod
    def emptyBackground():
        ZERO = np.array([0], dtype=np.float)

        parameters = (inputParameter.ne_numTimePts(ZERO),
                    inputParameter.ne_timePts(ZERO),
                    inputParameter.ne_paramType(ZERO),
                    inputParameter.ne_radCoord(ZERO),
                    inputParameter.ne_numInterpPts(ZERO),
                    inputParameter.ne_radGrid(ZERO),
                    inputParameter.ne_radGridPts(ZERO),
                    inputParameter.ne_decayLength(ZERO),
                    inputParameter.te_numTimePts(ZERO),
                    inputParameter.te_timePts(ZERO),
                    inputParameter.te_paramType(ZERO),
                    inputParameter.te_radCoord(ZERO),
                    inputParameter.te_numInterpPts(ZERO),
                    inputParameter.te_radGrid(ZERO),
                    inputParameter.te_radGridPts(ZERO),
                    inputParameter.te_decayLength(ZERO),
                    inputParameter.ti_numTimePts(ZERO),
                    inputParameter.ti_timePts(ZERO),
                    inputParameter.ti_paramType(ZERO),
                    inputParameter.ti_radCoord(ZERO),
                    inputParameter.ti_numInterpPts(ZERO),
                    inputParameter.ti_radGrid(ZERO),
                    inputParameter.ti_radGridPts(ZERO),
                    inputParameter.ti_decayLength(ZERO))

        return parameters

