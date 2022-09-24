import numpy as np

def checkIfIndicesInArrayBounds(iArg, jArg, kArg, arrayArg):
    return iArg < len(arrayArg)  and iArg >= 0 and jArg < len(arrayArg[0]) and jArg >= 0 and kArg < len(arrayArg[0][0]) and kArg >= 0


def computeDivergenceUFromDisplacementField(uArg, dxArg):
    divUOut = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
    uX = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
    uY = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
    uZ = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)


    for i in range(0, len(uArg)):
        for j in range(0, len(uArg[0])):
            for k in range(0, len(uArg[0][0])):
                uX[i][j][k] = uArg[i][j][k][0]
                uY[i][j][k] = uArg[i][j][k][1]
                uZ[i][j][k] = uArg[i][j][k][2]

    gradientUx = np.gradient(uX, dxArg, dxArg, dxArg, edge_order=2)
    gradientUy = np.gradient(uY, dxArg, dxArg, dxArg, edge_order=2)
    gradientUz = np.gradient(uZ, dxArg, dxArg, dxArg, edge_order=2)

    for i in range(0, len(uArg)):
        for j in range(0, len(uArg[0])):
            for k in range(0, len(uArg[0][0])):
                divUOut[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]

    return divUOut