import numpy as np
import copy


def zerothMoment(fArg):
    zerothMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0])))
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0,len(fArg[0][0])):
                zerothMomentOut[i][j][k] = fArg[i][j][k].sum()
    return zerothMomentOut


def firstMoment(fArg, ccArg):
    firstMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 3))
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0, len(fArg[0][0])):
                firstMomentOut[i][j][k] = np.zeros(3, dtype=np.double)
                for l in range(0, len(ccArg)):
                    firstMomentOut[i][j][k] = firstMomentOut[i][j][k] + fArg[i][j][k][l] * ccArg[l]
    return firstMomentOut


def secondMoment(fArg, ccArg):
    secondMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 3, 3), dtype=np.double)
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0, len(fArg[0][0])):
                secondMomentOut[i][j][k] = np.zeros((3, 3), dtype=np.double)
                for l in range(0, len(ccArg)):
                    secondMomentOut[i][j][k] = secondMomentOut[i][j][k] + fArg[i][j][k][l] * np.outer(ccArg[l], ccArg[l].transpose())
    return secondMomentOut


def sourceTerm(dxArg, rhoArg, rho0Arg, lamArg, mueArg, FArg):
    sourceTermOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 3), dtype=np.double)
    gradientRho = np.gradient(rhoArg, dxArg, dxArg, dxArg, axis=None,edge_order=2)
    for i in range(0, len(rhoArg)):
        for j in range(0, len(rhoArg[0])):
            for k in range(0, len(rhoArg[0][0])): # TODO divide by rho0
                #sourceTermOut[i][j][k] = FArg + 1.0/rhoArg[i][j][k]*(mueArg-lamArg)*np.array([gradientRho[0][i][j][k], gradientRho[1][i][j][k], gradientRho[2][i][j][k]])
                sourceTermOut[i][j][k] = FArg + 1.0 / rho0Arg * (mueArg - lamArg) * np.array(
                    [gradientRho[0][i][j][k], gradientRho[1][i][j][k], gradientRho[2][i][j][k]])
    return sourceTermOut


def equilibriumDistribution(rhoArg, jArg, PArg, ccArg, wArg, csArg):
    feqOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 27), dtype=np.double)
    for i in range(0, len(feqOut)):
        for j in range(0, len(feqOut[0])):
            for k in range(0, len(feqOut[0][0])):
                for l in range(0, len(feqOut[0][0][0])):
                    tmp2 = np.tensordot((PArg[i][j][k] - rhoArg[i][j][k] * csArg ** 2 * np.identity(3, dtype=np.double)), (np.outer(ccArg[l], ccArg[l].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)), axes=2)
                    tmp1 = np.tensordot(ccArg[l], jArg[i][j][k], axes=1)
                    feqOut[i][j][k][l] = wArg[l] * (rhoArg[i][j][k] + 1.0/(csArg ** 2) * tmp1 + 1.0 / (2.0 * csArg ** 4) * tmp2)
    return feqOut


import Util

def stream(fCollArg, ccArg , cArg):
   #fOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 27), dtype=np.double)
   fOut = copy.deepcopy(fCollArg)
   #np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 27), dtype=np.double)

   for i in range(0, len(fOut)):   # f0
       for j in range(0, len(fOut[0])):
           for k in range(0, len(fOut[0][0])):
               for l in range(0, len(fOut[0][0][0])):
                   cL = 1.0 / cArg * ccArg[l]
                   indicesToStreamTo = [int(i + cL[0]), int(j + cL[1]), int(k + cL[2])]
                   if Util.checkIfIndicesInArrayBounds(indicesToStreamTo[0], indicesToStreamTo[1], indicesToStreamTo[2], fOut):
                   #if indicesToStreamTo[0] < len(fOut) and indicesToStreamTo[1] < len(fOut[0]) and indicesToStreamTo[2] < len(fOut[0][0]):
                       fOut[indicesToStreamTo[0]][indicesToStreamTo[1]][indicesToStreamTo[2]][l] = fCollArg[i][j][k][l]
                   #else:
                       #print("hi")
   return fOut


def collide(fArg, feqArg, psiArg, dtArg, tauArg):
    # fCollOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), len(fArg[0][0][0])), dtype=np.double)
    fCollOut = fArg - dtArg/tauArg * (fArg - feqArg) + (1.0 - dtArg / (2.0 * tauArg)) * dtArg * psiArg
    return fCollOut

def sourceTermPsi(SArg, ccArg, wArg, csArg):
    psiOut = np.zeros((len(SArg), len(SArg[0]), len(SArg[0][0]), 27), dtype=np.double)
    for i in range(0, len(psiOut)):  # f0
        for j in range(0, len(psiOut[0])):
            for k in range(0, len(psiOut[0][0])):
                for l in range(0, len(psiOut[0][0][0])):
                    psiOut[i][j][k][l] = wArg[l] * 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], SArg[i][j][k], axes=1)
    return psiOut

def intitialize(rho0Arg, csArg, ccArg, wArg, mArg, nArg, oArg):
    P0 = np.zeros((mArg, nArg, oArg, 3, 3), dtype=np.double)
    j0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    u0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    rho = np.zeros((mArg, nArg, oArg), dtype=np.double)
    rho.fill(rho0Arg)
    fOut = equilibriumDistribution(rho, j0, P0, ccArg, wArg, csArg)
    return [fOut, j0, P0, u0]


def computeRho(fArg):
    return zerothMoment(fArg)


def computeJ(fArg,SArg,ccArg, dtArg):
    jOut = firstMoment(fArg, ccArg) + dtArg/2.0 * SArg
    return jOut

def computeP(fArg, ccArg):
    return secondMoment(fArg, ccArg)


def calculateMoments(fArg, SArg, ccArg, dtArg):
    return [computeRho(fArg), computeJ(fArg, SArg, ccArg, dtArg), computeP(fArg, ccArg)]


def computeU(uOldArg, rhoArg, jArg, jOldArg, dtArg, rho0Arg):
    uNew = np.zeros((len(uOldArg), len(uOldArg[0]), len(uOldArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(uOldArg)):  # f0
        for j in range(0, len(uOldArg[0])):
            for k in range(0, len(uOldArg[0][0])):
                #uNew[i][j][k] = uOldArg[i][j][k] + jArg[i][j][k] / rhoArg[i][j][k] * dtArg
                uNew[i][j][k] = uOldArg[i][j][k] + (jArg[i][j][k] + jOldArg[i][j][k] ) / rho0Arg / 2.0 * dtArg
    return uNew


# def computeDivergenceUFromDisplacementField(uArg, dxArg):
#     divUOut = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
#     uX = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
#     uY = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
#     uZ = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=np.double)
#
#
#     for i in range(0, len(uArg)):
#         for j in range(0, len(uArg[0])):
#             for k in range(0, len(uArg[0][0])):
#                 uX[i][j][k] = uArg[i][j][k][0]
#                 uY[i][j][k] = uArg[i][j][k][1]
#                 uZ[i][j][k] = uArg[i][j][k][2]
#
#     gradientUx = np.gradient(uX, dxArg, dxArg, dxArg, edge_order=2)
#     gradientUy = np.gradient(uY, dxArg, dxArg, dxArg, edge_order=2)
#     gradientUz = np.gradient(uZ, dxArg, dxArg, dxArg, edge_order=2)
#
#     for i in range(0, len(uArg)):
#         for j in range(0, len(uArg[0])):
#             for k in range(0, len(uArg[0][0])):
#                 divUOut[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]
#
#     return divUOut

def computeDivergenceUFromDensity(rhoArg,rho0Arg):
    divUOut = np.zeros(rhoArg.shape, dtype=np.double)
    for i in range(0,len(divUOut)):
        for j in range(0,len(divUOut[0])):
            for k in range(0,len(divUOut[0][0])):
                divUOut[i,j,k] = - (rhoArg[i,j,k] - rho0Arg) /rho0Arg
    return divUOut


def computeSigma(PArg, divUArg, laArg, mueArg):
    sigmaOut = np.zeros(PArg.shape, dtype=np.double) # TODO how is sigma computed?
    for i in range(0, len(sigmaOut)):
        for j in range(0, len(sigmaOut[0])):
            for k in range(0, len(sigmaOut[0][0])):
                sigmaOut[i,j,k] = -PArg[i, j, k] + (laArg-mueArg) * divUArg[i, j, k] * np.identity(3, dtype=np.double)
    return sigmaOut