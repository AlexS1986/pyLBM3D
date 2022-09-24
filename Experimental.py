import numpy as np
import copy


def equilibriumDistribution(rhoArg, jArg, sArg, ccArg, wArg, csArg, laArg, mueArg, rho0Arg):
    feqOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 27), dtype=np.double)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(feqOut)):
        for j in range(0, len(feqOut[0])):
            for k in range(0, len(feqOut[0][0])):
                for l in range(0, len(feqOut[0][0][0])):
                    tmp1 = np.tensordot(ccArg[l], jArg[i][j][k], axes=1)
                    tmp2 = np.tensordot(
                        (-sArg[i][j][k] - rhoArg[i][j][k] * csArg ** 2 * I),
                        (np.outer(ccArg[l], ccArg[l].transpose()) - csArg ** 2 * I),
                        axes=2)
                    leftTmp3 = np.einsum('a,bc->abc', jArg[i,j,k], I)
                    rightTmp3 = np.einsum('a,b,c->abc', ccArg[l], ccArg[l], ccArg[l]) - csArg ** 2 * (
                        np.einsum('a,bc->abc', ccArg[l], I) +  np.einsum('b,ac->abc', ccArg[l], I) +
                        np.einsum('c,ab->abc', ccArg[l], I))

                    tmp3 =  (np.einsum('abc,abc',leftTmp3,rightTmp3))
                    feqOut[i][j][k][l] = wArg[l] * (
                                rhoArg[i][j][k] + 1.0 / (csArg ** 2) * tmp1 + 1.0 / (2.0 * csArg ** 4) * tmp2 + 1.0 / (
                                    6.0 * csArg ** 6) * (laArg - mueArg) / rho0Arg * tmp3)

    return feqOut

def sourceTermPsi(bArg, rho0Arg, divJArg, ccArg, wArg, csArg, mueArg, laArg):
    psiOut = np.zeros((len(bArg), len(bArg[0]), len(bArg[0][0]), 27), dtype=np.double)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(psiOut)):
        for j in range(0, len(psiOut[0])):
            for k in range(0, len(psiOut[0][0])):
                for l in range(0, len(psiOut[0][0][0])):
                    left = divJArg[i,j,k] * I
                    right = np.outer(ccArg[l], ccArg[l].transpose()) - csArg ** 2 * I
                    tmp = np.einsum('ab,ab',left,right)
                    psiOut[i][j][k][l] = wArg[l] * rho0Arg, 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], bArg[i][j][k],
                                                                                              axes=1) + 1.0 / csArg ** 4 * (
                                                     mueArg - laArg) / rho0Arg * tmp

    return psiOut

def firstSource(bArg,rho0Arg):
    SOut = np.zeros((len(bArg), len(bArg[0]), len(bArg[0][0]), 3, 3), dtype=np.double)
    for i in range(0, len(SOut)):
        for j in range(0, len(SOut[0])):
            for k in range(0, len(SOut[0][0])):
                SOut[i,j,k] = bArg[i,j,k] * rho0Arg
    return SOut

def secondSource(divJArg,laArg,mueArg,rho0Arg):
    SOut = np.zeros((len(divJArg), len(divJArg[0]), len(divJArg[0][0]), 3), dtype=np.double)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(SOut)):
        for j in range(0, len(SOut[0])):
            for k in range(0, len(SOut[0][0])):
                SOut[i, j, k] = (mueArg - laArg) / rho0Arg * divJArg[i,j,k] * I
    return SOut

def rho(zerothMomentArg):
    return copy.deepcopy(zerothMomentArg)

def j(firstMomentArg,dtArg,firstSourceArg):
    jOut = np.zeros(firstMomentArg.shape, dtype=np.double)
    for i in range(0, len(jOut)):
        for j in range(0, len(jOut[0])):
            for k in range(0, len(jOut[0][0])):
                jOut[i,j,k] = firstMomentArg[i,j,k] + dtArg/2.0 * firstSourceArg[i,j,k]
    return jOut

def sigma(secondMomentArg,dtArg,secondSourceArg):
    sigmaOut = np.zeros(secondMomentArg.shape, dtype=np.double)
    for i in range(0, len(sigmaOut)):
        for j in range(0, len(sigmaOut[0])):
            for k in range(0, len(sigmaOut[0][0])):
                sigmaOut[i,j,k] = secondMomentArg[i,j,k] + dtArg/2.0 * secondSourceArg[i,j,k]
    return sigmaOut


def computeU(uOldArg, rho0Arg, jArg, jOldArg, dtArg):
    uNew = np.zeros((len(uOldArg), len(uOldArg[0]), len(uOldArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(uOldArg)):  # f0
        for j in range(0, len(uOldArg[0])):
            for k in range(0, len(uOldArg[0][0])):
                uNew[i][j][k] = uOldArg[i][j][k] + (jArg[i][j][k] + jOldArg[i][j][k]) / rho0Arg / 2.0 * dtArg
    return uNew