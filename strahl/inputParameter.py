
class inputParameter():
    """docstring for inputParameter"""

    def __init__(self, name, priority, value):
        self.name = name
        self.priority = priority
        self.value = value
        self.shape = value.shape
        self.size = value.size

    def newValue(self, newValue):
        self.value = newValue
        self.shape = newValue.shape
        self.size = newValue.size

# Main Input Parameters -- at the moment this is limited
    @classmethod
    def atomic_weight(cls, value):
        return cls("atomic_weight", 1, value)

    @classmethod
    def charge(cls, value):
        return cls("charge", 2, value)

# Background Input Parameters
    @classmethod
    def ne_numTimePts(cls, value):
        return cls("ne_numTimePts", 1, value)

    @classmethod
    def ne_timePts(cls, value):
        return cls("ne_timePts", 2, value)

    @classmethod
    def ne_paramType(cls, value):
        return cls("ne_paramType", 3, value)

    @classmethod
    def ne_radCoord(cls, value):
        return cls("ne_radCoord", 4, value)

    @classmethod
    def ne_numInterpPts(cls, value):
        return cls("ne_numInterpPts", 5, value)

    @classmethod
    def ne_radGrid(cls, value):
        return cls("ne_radGrid", 6, value)

    @classmethod
    def ne_radGridPts(cls, value):
        return cls("ne_radGridPts", 7, value)

    @classmethod
    def ne_decayLength(cls, value):
        return cls("ne_decayLength", 8, value)

    @classmethod
    def te_numTimePts(cls, value):
        return cls("te_numTimePts", 9, value)

    @classmethod
    def te_timePts(cls, value):
        return cls("te_timePts", 10, value)

    @classmethod
    def te_paramType(cls, value):
        return cls("te_paramType", 11, value)

    @classmethod
    def te_radCoord(cls, value):
        return cls("te_radCoord", 12, value)

    @classmethod
    def te_numInterpPts(cls, value):
        return cls("te_numInterpPts", 13, value)

    @classmethod
    def te_radGrid(cls, value):
        return cls("te_radGrid", 14, value)

    @classmethod
    def te_radGridPts(cls, value):
        return cls("te_radGridPts", 15, value)

    @classmethod
    def te_decayLength(cls, value):
        return cls("te_decayLength", 16, value)

    @classmethod
    def ti_numTimePts(cls, value):
        return cls("ti_numTimePts", 17, value)

    @classmethod
    def ti_timePts(cls, value):
        return cls("ti_timePts", 18, value)

    @classmethod
    def ti_paramType(cls, value):
        return cls("ti_paramType", 19, value)

    @classmethod
    def ti_radCoord(cls, value):
        return cls("ti_radCoord", 20, value)

    @classmethod
    def ti_numInterpPts(cls, value):
        return cls("ti_numInterpPts", 21, value)

    @classmethod
    def ti_radGrid(cls, value):
        return cls("ti_radGrid", 22, value)

    @classmethod
    def ti_radGridPts(cls, value):
        return cls("ti_radGridPts", 23, value)

    @classmethod
    def ti_decayLength(cls, value):
        return cls("ti_decayLength", 24, value)

# Geometry Input Parameters

# Flux Input Parameters

