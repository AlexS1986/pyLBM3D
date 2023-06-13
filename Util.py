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

def computeGradientU(uArg, dxArg): # TODO different stencil 
    uX = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
    uY = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
    uZ = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)

    gradU = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0]), 3, 3), dtype=np.double)

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
                gradU[i][j][k] = [ [gradientUx[0][i][j][k], gradientUx[1][i][j][k], gradientUx[2][i][j][k]],
                                   [gradientUy[0][i][j][k], gradientUy[1][i][j][k], gradientUy[2][i][j][k]],
                                   [gradientUz[0][i][j][k], gradientUz[1][i][j][k], gradientUz[2][i][j][k]]]
                #gradU[i][j][k] = np.gradient(uArg[i,j,k],dxArg, dxArg, dxArg, edge_order=2)

    return gradU


def dJyDy(jArg, dxArg):
    j1d1 = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)
    j2d2 = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)
    j3d3 = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)
    uX = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)
    uY = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)
    uZ = np.zeros((len(jArg), len(jArg[0]), len(jArg[0][0])), dtype=np.double)


    for i in range(0, len(jArg)):
        for j in range(0, len(jArg[0])):
            for k in range(0, len(jArg[0][0])):
                uX[i][j][k] = jArg[i][j][k][0]
                uY[i][j][k] = jArg[i][j][k][1]
                uZ[i][j][k] = jArg[i][j][k][2]

    gradientUx = np.gradient(uX, dxArg, dxArg, dxArg, edge_order=2)
    gradientUy = np.gradient(uY, dxArg, dxArg, dxArg, edge_order=2)
    gradientUz = np.gradient(uZ, dxArg, dxArg, dxArg, edge_order=2)

    for i in range(0, len(jArg)):
        for j in range(0, len(jArg[0])):
            for k in range(0, len(jArg[0][0])):
                j1d1[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]
                j2d2[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]
                j3d3[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]

    return [j1d1, j2d2, j3d3]

def trace(eps):
    trEps = np.zeros((len(eps), len(eps[0]), len(eps[0][0])), dtype=np.double)
    for i in range(0, len(eps)):
        for j in range(0, len(eps[0])):
            for k in range(0, len(eps[0][0])):
                trEps[i,j,k] = eps[i,j,k,0,0] +  eps[i,j,k,1,1] + eps[i,j,k,2,2]
    return trEps