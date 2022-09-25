import numpy as np
import copy


def intitialize(rho0Arg, csArg, ccArg, wArg, mArg, nArg, oArg, lamArg, mueArg):
    sigma0 = np.zeros((mArg, nArg, oArg, 3, 3), dtype=np.double)
    j0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    u0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    rho = np.zeros((mArg, nArg, oArg), dtype=np.double)
    rho.fill(rho0Arg)
    fOut = equilibriumDistribution(rho, j0, sigma0, ccArg, wArg, csArg, lamArg, mueArg, rho0Arg)
    return [fOut, j0, sigma0, u0]


def equilibriumDistribution(rhoArg, jArg, sigmaArg, ccArg, wArg, csArg, lamArg, mueArg, rho0Arg):
    feqOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 27), dtype=np.double)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(feqOut)):
        for j in range(0, len(feqOut[0])):
            for k in range(0, len(feqOut[0][0])):
                for l in range(0, len(feqOut[0][0][0])):
                    tmp1 = np.tensordot(ccArg[l], jArg[i][j][k], axes=1)
                    tmp2 = np.tensordot(
                        (-sigmaArg[i][j][k] - rhoArg[i][j][k] * csArg ** 2 * I),
                        (np.outer(ccArg[l], ccArg[l].transpose()) - csArg ** 2 * I),
                        axes=2)
                    leftTmp3 = np.einsum('a,bc->abc', jArg[i,j,k], I)
                    rightTmp3 = np.einsum('a,b,c->abc', ccArg[l], ccArg[l], ccArg[l]) - csArg ** 2 * (
                        np.einsum('a,bc->abc', ccArg[l], I) +  np.einsum('b,ac->abc', ccArg[l], I) +
                        np.einsum('c,ab->abc', ccArg[l], I))

                    tmp3 =  (np.einsum('abc,abc',leftTmp3,rightTmp3))
                    feqOut[i][j][k][l] = wArg[l] * (
                                rhoArg[i][j][k] + 1.0 / (csArg ** 2) * tmp1 + 1.0 / (2.0 * csArg ** 4) * tmp2 + 1.0 / (
                                    6.0 * csArg ** 6) * (lamArg - mueArg) / rho0Arg * tmp3)

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
                    #test = wArg[l] * rho0Arg, 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], bArg[i][j][k], axes=1) + 1.0 / (csArg ** 4) * (mueArg - laArg) / rho0Arg * tmp
                    psiOut[i][j][k][l] = wArg[l] * (rho0Arg * 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], bArg[i][j][k], axes=1) + 1.0 / (csArg ** 4) * (mueArg - laArg) / rho0Arg * tmp)
    return psiOut

def firstSource(bArg,rho0Arg):
    SOut = np.zeros((len(bArg), len(bArg[0]), len(bArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(SOut)):
        for j in range(0, len(SOut[0])):
            for k in range(0, len(SOut[0][0])):
                SOut[i,j,k] = bArg[i,j,k] * rho0Arg
    return SOut

def secondSource(divJArg,laArg,mueArg,rho0Arg):
    SOut = np.zeros((len(divJArg), len(divJArg[0]), len(divJArg[0][0]), 3, 3), dtype=np.double)
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


import BoundaryConditions as BC


def applyNeumannBoundaryConditions(fArg, fCollArg, uArg , rhoArg, rho0Arg, csArg, ccArg, cArg, wArg, sigmaBdArg, sigmaArg, dxArg, laArg, mueArg, coordinateArg='x', coordinateValueArg=0):
    rhoBd = BC.computeRhoBdWithoutExtrapolation(rhoArg,ccArg,coordinateArg,coordinateValueArg)
    fCollRelevant = BC.selectAtCoordinate(fCollArg, coordinateArg, coordinateValueArg)
    fOut = copy.deepcopy(fArg)
    fRelevant = BC.selectAtCoordinate(fOut, coordinateArg, coordinateValueArg)
    indicesMissing = BC.getMissingDistributionFunctionIndices(fArg, coordinateArg, coordinateValueArg)
    sigmaBd = computeSigmaBd(sigmaBdArg, sigmaArg, fArg, ccArg, cArg, uArg, dxArg, laArg, mueArg, rhoArg, rho0Arg, coordinateArg='x', coordinateValueArg=0)
    #PBd = computePBd(sigmaBdArg,fArg,ccArg,cArg,uArg,dxArg,laArg,mueArg,rhoArg, rho0Arg, coordinateArg,coordinateValueArg)
    for i in range(0, len(fRelevant)):
        for j in range(0, len(fRelevant[0])):
            for l in indicesMissing:
                oL = BC.getOppositeLatticeDirection(l)

                tmp1 = -sigmaBd[i,j,l] - rhoBd[i,j,l] * csArg ** 2 * np.identity(3,dtype=np.double)
                tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
                fRelevant[i,j,l] = - fCollRelevant[i, j, oL] + 2.0 * wArg[oL] * (rhoBd[i,j,l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))

    return fOut


def applyNeumannBoundaryConditionsAtEdge(fArg, fCollArg, uArg , rhoArg, rho0Arg, csArg, ccArg, cArg, wArg, sigmaBdArg1, sigmaBdArg2, sigmaArg, dxArg, laArg, mueArg, coordinateArg1='x', coordinateValueArg1=0, coordinateArg2='y', coordinateValueArg2=0):
    rhoBd = BC.reduceSurfaceToEdge(BC.computeRhoBdWithoutExtrapolation(rhoArg, ccArg, coordinateArg1, coordinateValueArg1),coordinateArg1,coordinateArg2,coordinateValueArg2)
    fCollRelevant = BC.selectAtEdge(fCollArg,coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    fOut = copy.deepcopy(fArg)
    fRelevant = BC.selectAtEdge(fOut,coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    indicesMissing = BC.getMissingDistributionFunctionIndicesAtEdge(fArg, coordinateArg1,coordinateValueArg1, coordinateArg2, coordinateValueArg2)

    sigmaBd = 1.0/2.0 * (sigmaBdArg1 + sigmaBdArg2)
    sigmaBd = BC.reduceSurfaceToEdge(computeSigmaBd(sigmaBd, sigmaArg, fArg, ccArg, cArg, uArg, dxArg, laArg, mueArg, rhoArg, rho0Arg, coordinateArg='x', coordinateValueArg=0), coordinateArg1,coordinateArg2,coordinateValueArg2)

    for i in range(0, len(fRelevant)):
        for l in indicesMissing:
            oL = BC.getOppositeLatticeDirection(l)

            tmp1 = -sigmaBd[i,l] - rhoBd[i,l] * csArg ** 2 * np.identity(3,dtype=np.double)
            tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
            #print(fRelevant.shape)
            fRelevant[i,l] = - fCollRelevant[i,  oL] + 2.0 * wArg[oL] * (rhoBd[i,l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))

    return fOut


def applyNeumannBoundaryConditionsAtCorner(fArg, fCollArg, uArg , rhoArg, rho0Arg, csArg, ccArg, cArg, wArg, sigmaBdArg1, sigmaBdArg2, sigmaBdArg3, sigmaArg, dxArg, laArg, mueArg, coordinateValueArg1=0, coordinateValueArg2=0, coordinateValueArg3=0):
    rhoBd = BC.reduceSurfaceToCorner(BC.computeRhoBdWithoutExtrapolation(rhoArg, ccArg, 'x', coordinateValueArg1),coordinateValueArg2, coordinateValueArg3)
    #rhoBd = selectAtEdge(rhoArg, coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    fCollRelevant = fCollArg[coordinateValueArg1, coordinateValueArg2,coordinateValueArg3] #selectAtEdge(fCollArg,coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    fOut = copy.deepcopy(fArg)
    fRelevant = fOut[coordinateValueArg1, coordinateValueArg2,coordinateValueArg3]
    indicesMissing = BC.getMissingDistributionFunctionIndicesAtCorner(fArg, coordinateValueArg1, coordinateValueArg2, coordinateValueArg3)

    sigmaBd = 1.0/3.0 * (sigmaBdArg1 + sigmaBdArg2 + sigmaBdArg3) # TODO average here okay?
    sigmaBd = BC.reduceSurfaceToCorner(computeSigmaBd(sigmaBd, sigmaArg, fArg, ccArg, cArg, uArg, dxArg, laArg, mueArg, rhoArg, rho0Arg, 'x',
                    coordinateValueArg1), coordinateValueArg2, coordinateValueArg3)

    for l in indicesMissing:
        oL = BC.getOppositeLatticeDirection(l)

        tmp1 = -sigmaBd[l] - rhoBd[l] * csArg ** 2 * np.identity(3,dtype=np.double)
        tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
        fRelevant[l] = - fCollRelevant[ oL] + 2.0 * wArg[oL] * (rhoBd[l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))

    return fOut

import Core

def computeSigmaBd(sigmaBC, sigmaArg, fArg, ccArg, cArg, uArg, dxArg, laArg, mueArg, rhoArg, rho0Arg, coordinateArg='x', coordinateValueArg=0):
    def applyTractionBoundaryConditions(sigmaBd, sigmaBC):
        sigmaBd = copy.deepcopy(sigmaBd)
        for i in range(0, len(sigmaBd)):
            for j in range(0, len(sigmaBd[i])):
                for l in range(0, len(sigmaBd[i][j])):
                    for ii in range(0, len(sigmaBd[i][j][l])):
                        for jj in range(0, len(sigmaBd[i][j][l][ii])):
                            if (not np.isnan(sigmaBC[ii,jj])):
                                sigmaBd[i,j,l,ii,jj] = sigmaBC[ii,jj]
        return sigmaBd

    def computeSigmaBdWithoutExtrapolation(sigmaArg, ccArg, coordinateArg,coordinateValueArg):
        sigmaAtCoordinate = BC.selectAtCoordinate(sigmaArg,coordinateArg,coordinateValueArg)
        sigmaBd = np.zeros((len(sigmaArg),len(sigmaArg[0]),len(ccArg),3,3), dtype=np.double)

        for i in range(0, len(sigmaBd)):
            for j in range(0, len(sigmaBd[i])):
                for l in range(0, len(sigmaBd[i][j])):
                    sigmaBd[i,j,l] = sigmaAtCoordinate[i,j]
        return sigmaBd

    sigmaBd = applyTractionBoundaryConditions(computeSigmaBdWithoutExtrapolation(sigmaArg,ccArg,coordinateArg,coordinateValueArg), sigmaBC) # without extrapolation
    return sigmaBd