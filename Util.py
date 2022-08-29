def checkIfIndicesInArrayBounds(iArg, jArg, kArg, arrayArg):
    return iArg < len(arrayArg)  and iArg >= 0 and jArg < len(arrayArg[0]) and jArg >= 0 and kArg < len(arrayArg[0][0]) and kArg >= 0