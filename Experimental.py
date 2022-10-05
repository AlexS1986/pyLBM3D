import numpy as np
import copy


def intitialize(rho0Arg, csArg, ccArg, wArg, mArg, nArg, oArg, lamArg, mueArg):
    '''
    returns the initial distribution functions (as equilibrium distributions), moments and displacement field
    :param rho0Arg: the initial density, a scalar
    :param csArg: wave speed of shear waves, a scalar
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param mArg: dimension of lattice in x direction
    :param nArg: dimension of lattice in y direction
    :param oArg: dimension of lattice in z direction
    :param lamArg: Lame parameter
    :param mueArg: Lame parameter
    :return: [fOut, j0, sigma0, u0]
    '''
    sigma0 = np.zeros((mArg, nArg, oArg, 3, 3), dtype=np.double)
    j0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    u0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    rho = np.zeros((mArg, nArg, oArg), dtype=np.double)
    rho.fill(rho0Arg)
    fOut = equilibriumDistribution(rho, j0, sigma0, ccArg, wArg, csArg, lamArg, mueArg, rho0Arg)
    return [fOut, j0, sigma0, u0]


def equilibriumDistribution(rhoArg, jArg, sigmaArg, ccArg, wArg, csArg, lamArg, mueArg, rho0Arg):
    '''
    :param rhoArg: the density in lattice dimensions (m,n,o)
    :param jArg: the momentum in lattice dimensions (m,n,o,3)
    :param sigmaArg: the stress in lattice dimensions (m,n,o,3,3)
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param csArg: wave speed of shear waves, a scalar
    :param lamArg: Lame parameter
    :param mueArg: Lame parameter
    :param rho0Arg: the initial density, a scalar
    :return: feqOut the equilibrium distribution in lattice dimensions (m,n,o)
    '''
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
                    leftTmp3 = np.einsum('a,bc->abc', jArg[i, j, k], I)
                    rightTmp3 = np.einsum('a,b,c->abc', ccArg[l], ccArg[l], ccArg[l]) - csArg ** 2 * (
                        np.einsum('a,bc->abc', ccArg[l], I) + np.einsum('b,ac->abc', ccArg[l], I) +
                        np.einsum('c,ab->abc', ccArg[l], I))

                    tmp3 = (np.einsum('abc,abc', leftTmp3, rightTmp3))
                    feqOut[i][j][k][l] = wArg[l] * (
                                rhoArg[i][j][k] + 1.0 / (csArg ** 2) * tmp1 + 1.0 / (2.0 * csArg ** 4) * tmp2 + 1.0 / (
                                    6.0 * csArg ** 6) * (lamArg - mueArg) / rho0Arg * tmp3)

    return feqOut


def sourceTermPsi(bArg, rho0Arg, dJyDy, ccArg, wArg, csArg, mueArg, laArg):
    '''
    :param bArg: the body force per unit mass in lattice dimensions (m,n,o,3)
    :param rho0Arg: the original density, a scalar
    :param dJyDy: an array of the derivatives of the momentum [dJxDx, dJyDy, dJzDz] in lattice dimensions (3,m,n,o)
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param csArg: wave speed of shear waves, a scalar
    :param mueArg: Lame parameter
    :param laArg: Lame parameter
    :return: the source term in the LBE in lattice dimensions (m,n,o,27)
    '''
    psiOut = np.zeros((len(bArg), len(bArg[0]), len(bArg[0][0]), 27), dtype=np.double)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(psiOut)):
        for j in range(0, len(psiOut[0])):
            for k in range(0, len(psiOut[0][0])):
                for l in range(0, len(psiOut[0][0][0])):
                    left = np.array([[dJyDy[0][i, j, k], 0, 0], [0, dJyDy[1][i, j, k], 0], [0, 0, dJyDy[2][i, j, k]]], dtype=float)
                    right = np.outer(ccArg[l], ccArg[l].transpose()) - csArg ** 2 * I
                    tmp = np.einsum('ab,ab', left, right)
                    psiOut[i][j][k][l] = wArg[l] * (rho0Arg * 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], bArg[i][j][k], axes=1) + 1.0 / (csArg ** 4) * (mueArg - laArg) / rho0Arg * tmp)
    return psiOut

def firstSource(bArg, rho0Arg):
    '''
    :param bArg: the body force per unit mass in lattice dimensions (m,n,o,3)
    :param rho0Arg: the original density, a scalar
    :return: the source term in the first moment (m,n,o,3)
    '''
    SOut = np.zeros((len(bArg), len(bArg[0]), len(bArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(SOut)):
        for j in range(0, len(SOut[0])):
            for k in range(0, len(SOut[0][0])):
                SOut[i,j,k] = bArg[i,j,k] * rho0Arg
    return SOut

def secondSource(dJyDy, laArg, mueArg, rho0Arg):
    '''
    :param dJyDy: an array of the derivatives of the momentum [dJxDx, dJyDy, dJzDz] in lattice dimensions (3,m,n,o)
    :param laArg: Lame parameter
    :param mueArg: Lame parameter
    :param rho0Arg: the original density, a scalar
    :return: the source term in the second moment (m,n,o,3,3)
    '''
    SOut = np.zeros((len(dJyDy[0]), len(dJyDy[0][0]), len(dJyDy[0][0][0]), 3, 3), dtype=np.double)
    #I = np.identity(3, dtype=np.double)
    for i in range(0, len(SOut)):
        for j in range(0, len(SOut[0])):
            for k in range(0, len(SOut[0][0])):
                SOut[i, j, k] = (mueArg - laArg) / rho0Arg * np.array([[dJyDy[0][i,j,k], 0, 0], [0, dJyDy[1][i,j,k], 0], [0, 0, dJyDy[2][i,j,k]]], dtype=float)
    return SOut


def rho(zerothMomentArg):
    '''
    :param zerothMomentArg: the zeroth moment in lattice dimensions (m,n,o)
    :return: the current density in lattice dimensions (m,n,o)
    '''
    return copy.deepcopy(zerothMomentArg)


def j(firstMomentArg,dtArg,firstSourceArg):
    '''
    :param firstMomentArg: the first moment in lattice dimensions (m,n,o,3)
    :param dtArg: the time step size, a scalar
    :param firstSourceArg: the first order source term in lattice dimensions (m,n,o,3)
    :return: the momentum in lattice dimensions (m,n,o,3)
    '''
    jOut = np.zeros(firstMomentArg.shape, dtype=np.double)
    for i in range(0, len(jOut)):
        for j in range(0, len(jOut[0])):
            for k in range(0, len(jOut[0][0])):
                jOut[i,j,k] = firstMomentArg[i,j,k] + dtArg/2.0 * firstSourceArg[i,j,k]
    return jOut


def sigma(secondMomentArg,dtArg,secondSourceArg):
    '''
    :param secondMomentArg: the second moment in lattice dimensions (m,n,o,3,3)
    :param dtArg: the time step size, a scalar
    :param secondSourceArg: the second order source term in lattice dimensions (m,n,o,3,3)
    :return: the Cauchy stress in lattice dimensions (m,n,o,3,3)
    '''
    sigmaOut = np.zeros(secondMomentArg.shape, dtype=np.double)
    for i in range(0, len(sigmaOut)):
        for j in range(0, len(sigmaOut[0])):
            for k in range(0, len(sigmaOut[0][0])):
                sigmaOut[i,j,k] = -secondMomentArg[i,j,k] - dtArg/2.0 * secondSourceArg[i,j,k]
    return sigmaOut


def computeU(uOldArg, rho0Arg, jArg, jOldArg, dtArg):
    '''
    :param uOldArg: the displacement from the last time step in lattice dimensions (m,n,o,3)
    :param rho0Arg: the original density, a scalar
    :param jArg: the current momentum in lattice dimensions (m,n,o,3)
    :param jOldArg: the momentum of the last time step in lattice dimensions (m,n,o,3)
    :param dtArg: the time step size
    :return: the displacement for the current time step in lattice dimensions (m,n,o,3)
    '''
    uNew = np.zeros((len(uOldArg), len(uOldArg[0]), len(uOldArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(uOldArg)):  # f0
        for j in range(0, len(uOldArg[0])):
            for k in range(0, len(uOldArg[0][0])):
                uNew[i][j][k] = uOldArg[i][j][k] + (jArg[i][j][k] + jOldArg[i][j][k]) / rho0Arg / 2.0 * dtArg
    return uNew


import BoundaryConditions as BC


def neumannBoundaryRule(sigmaBdArg, rhoBdArg, csArg, ccArg, wArg, fCollRelevantArg):
    tmp1 = -sigmaBdArg - rhoBdArg * csArg ** 2 * np.identity(3, dtype=np.double)
    tmp2 = np.outer(ccArg, ccArg.transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
    fBouncedBack = - fCollRelevantArg + 2.0 * wArg * (rhoBdArg + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1, tmp2, axes=2)))
    return fBouncedBack



def applyNeumannBoundaryConditions(fArg, fCollArg, rhoArg, csArg, ccArg, wArg, sigmaBdArg, sigmaArg, coordinateArg='x', coordinateValueArg=0, boundaryRule = neumannBoundaryRule):
    '''
    :param fArg: the distribution function before the boundary conditions have been applied at the given plane in lattice dimensions (m,n,o)
    :param fCollArg: the distribution function after collision has been applied in lattice dimensions (m,n,o)
    :param rhoArg: the density in lattice dimensions (m,n,o)
    :param csArg: csArg: wave speed of shear waves, a scalar
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param sigmaBdArg: the prescribed stress at this plane with nan's for undefined values (3,3)
    :param sigmaArg: the stress field in lattice dimensions (m,n,o,3,3)
    :param coordinateArg: 'x', 'y', 'z' the coordinate direction identifying the plane
    :param coordinateValueArg: the index identifying the plane (either 0 or max in the respective direction)
    :return: the distribution function after the boundary conditions have been applied at the given plane in lattice dimensions (m,n,o)
    '''
    rhoBd = BC.computeRhoBdWithoutExtrapolation(rhoArg, ccArg, coordinateArg, coordinateValueArg)
    fCollRelevant = BC.selectAtCoordinate(fCollArg, coordinateArg, coordinateValueArg)
    fOut = copy.deepcopy(fArg)
    fRelevant = BC.selectAtCoordinate(fOut, coordinateArg, coordinateValueArg)
    indicesMissing = BC.getMissingDistributionFunctionIndices(fArg, coordinateArg, coordinateValueArg)
    sigmaBd = computeSigmaBd(sigmaBdArg, sigmaArg, ccArg, coordinateArg='x', coordinateValueArg=0)
    for i in range(0, len(fRelevant)):
        for j in range(0, len(fRelevant[0])):
            for l in indicesMissing:
                oL = BC.getOppositeLatticeDirection(l)

                #tmp1 = -sigmaBd[i, j, l] - rhoBd[i, j, l] * csArg ** 2 * np.identity(3, dtype=np.double)
                #tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
                #fRelevant[i, j, l] = - fCollRelevant[i, j, oL] + 2.0 * wArg[oL] * (rhoBd[i, j, l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1, tmp2, axes=2)))
                fRelevant[i, j, l] = boundaryRule(sigmaBd[i, j, l], rhoBd[i, j, l], csArg, ccArg[oL], wArg[oL], fCollRelevant[i, j, oL])

    return fOut


def applyNeumannBoundaryConditionsAtEdge(fArg, fCollArg,  rhoArg,  csArg, ccArg,  wArg, sigmaBdArg1, sigmaBdArg2, sigmaArg, coordinateArg1='x', coordinateValueArg1=0, coordinateArg2='y', coordinateValueArg2=0, boundaryRule=neumannBoundaryRule):
    '''

    :param fArg: the distribution function before the boundary conditions have been applied at the given plane in lattice dimensions (m,n,o)
    :param fCollArg: the distribution function after collision has been applied in lattice dimensions (m,n,o)
    :param rhoArg: the density in lattice dimensions (m,n,o)
    :param csArg: csArg: wave speed of shear waves, a scalar
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param sigmaBdArg1: the prescribed stress at this plane with nan's for undefined values (3,3) at the first plane
    :param sigmaBdArg2: the prescribed stress at this plane with nan's for undefined values (3,3) at the second plane
    :param sigmaArg: the stress field in lattice dimensions (m,n,o,3,3)
    :param coordinateArg1: 'x', 'y', 'z' the coordinate direction identifying the plane 1
    :param coordinateValueArg1: the index identifying the plane 1 (either 0 or max in the respective direction)
    :param coordinateArg2: 'x', 'y', 'z' the coordinate direction identifying the plane 2
    :param coordinateValueArg2: the index identifying the plane 2 (either 0 or max in the respective direction)
    :return: the distribution function after the boundary conditions have been applied at the given edge in lattice dimensions (m,n,o)
    '''
    rhoBd = BC.reduceSurfaceToEdge(BC.computeRhoBdWithoutExtrapolation(rhoArg, ccArg, coordinateArg1, coordinateValueArg1),coordinateArg1,coordinateArg2,coordinateValueArg2)
    fCollRelevant = BC.selectAtEdge(fCollArg,coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    fOut = copy.deepcopy(fArg)
    fRelevant = BC.selectAtEdge(fOut, coordinateArg1, coordinateValueArg1, coordinateArg2, coordinateValueArg2)
    indicesMissing = BC.getMissingDistributionFunctionIndicesAtEdge(fArg, coordinateArg1,coordinateValueArg1, coordinateArg2, coordinateValueArg2)

    sigmaBd = 1.0/2.0 * (sigmaBdArg1 + sigmaBdArg2)
    sigmaBd = BC.reduceSurfaceToEdge(computeSigmaBd(sigmaBd, sigmaArg, ccArg, coordinateArg='x', coordinateValueArg=0), coordinateArg1,coordinateArg2,coordinateValueArg2)

    for i in range(0, len(fRelevant)):
        for l in indicesMissing:
            oL = BC.getOppositeLatticeDirection(l)

            #tmp1 = -sigmaBd[i, l] - rhoBd[i, l] * csArg ** 2 * np.identity(3,dtype=np.double)
            #tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
            #fRelevant[i,l] = - fCollRelevant[i,  oL] + 2.0 * wArg[oL] * (rhoBd[i,l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))

            fRelevant[i, l] = boundaryRule(sigmaBd[i,  l], rhoBd[i,  l], csArg, ccArg[oL], wArg[oL], fCollRelevant[i, oL])
    return fOut


def applyNeumannBoundaryConditionsAtCorner(fArg, fCollArg, rhoArg,  csArg, ccArg,  wArg, sigmaBdArg1, sigmaBdArg2, sigmaBdArg3, sigmaArg,  coordinateValueArg1=0, coordinateValueArg2=0, coordinateValueArg3=0, boundaryRule=neumannBoundaryRule):
    '''

    :param fArg: the distribution function before the boundary conditions have been applied at the given plane in lattice dimensions (m,n,o)
    :param fCollArg: the distribution function after collision has been applied in lattice dimensions (m,n,o)
    :param rhoArg: the density in lattice dimensions (m,n,o)
    :param csArg: csArg: wave speed of shear waves, a scalar
    :param ccArg: array of lattice speeds, (27,3)
    :param wArg: weights of lattice directions (27)
    :param sigmaBdArg1: the prescribed stress at this plane with nan's for undefined values (3,3) at the first plane, x-direction
    :param sigmaBdArg2: the prescribed stress at this plane with nan's for undefined values (3,3) at the second plane, y-direction
    :param sigmaBdArg3: the prescribed stress at this plane with nan's for undefined values (3,3) at the third plane, z-direction
    :param sigmaArg: the stress field in lattice dimensions (m,n,o,3,3)
    :param coordinateValueArg1: the index identifying the plane 1 (either 0 or max in the x direction)
    :param coordinateValueArg2: the index identifying the plane 2 (either 0 or max in the y direction)
    :param coordinateValueArg3: the index identifying the plane 3 (either 0 or max in the z direction)
    :return: the distribution function after the boundary conditions have been applied at the given corner in lattice dimensions (m,n,o)
    '''
    rhoBd = BC.reduceSurfaceToCorner(BC.computeRhoBdWithoutExtrapolation(rhoArg, ccArg, 'x', coordinateValueArg1),coordinateValueArg2, coordinateValueArg3)
    fCollRelevant = fCollArg[coordinateValueArg1, coordinateValueArg2, coordinateValueArg3]
    fOut = copy.deepcopy(fArg)
    fRelevant = fOut[coordinateValueArg1, coordinateValueArg2, coordinateValueArg3]
    indicesMissing = BC.getMissingDistributionFunctionIndicesAtCorner(fArg, coordinateValueArg1, coordinateValueArg2, coordinateValueArg3)

    sigmaBd = 1.0/3.0 * (sigmaBdArg1 + sigmaBdArg2 + sigmaBdArg3) # TODO average here okay?
    sigmaBd = BC.reduceSurfaceToCorner(
        computeSigmaBd(sigmaBd, sigmaArg, ccArg, coordinateArg='x', coordinateValueArg=coordinateValueArg1), coordinateValueArg2, coordinateValueArg3)

    for l in indicesMissing:
        oL = BC.getOppositeLatticeDirection(l)

        #tmp1 = -sigmaBd[l] - rhoBd[l] * csArg ** 2 * np.identity(3,dtype=np.double)
        #tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=np.double)
        #fRelevant[l] = - fCollRelevant[oL] + 2.0 * wArg[oL] * (rhoBd[l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))
        fRelevant[l] = boundaryRule(sigmaBd[l], rhoBd[l], csArg, ccArg[oL], wArg[oL], fCollRelevant[oL])

    return fOut


def computeSigmaBd(sigmaBC, sigmaArg, ccArg, coordinateArg='x', coordinateValueArg=0):
    '''

    :param sigmaBC: the prescribed stress at this plane with nan's for undefined values (3,3)
    :param sigmaArg:  the stress field in lattice dimensions (m,n,o,3,3)
    :param ccArg: array of lattice speeds, (27,3)
    :param coordinateArg: 'x', 'y', 'z' the coordinate direction identifying the plane
    :param coordinateValueArg: the index identifying the plane (either 0 or max in the respective direction)
    :return: the stress field for all lattice links at all lattice points of a given plane, accounting for boundary conditions; in plane dimensions (k,l,27,3,3)
    '''
    def applyTractionBoundaryConditions(sigmaBd, sigmaBC):
        '''

        :param sigmaBd: the current stress field at the plane in plane dimensions (k,l,27,3,3)
        :param sigmaBC: the prescribed stress at this plane with nan's for undefined values (3,3)
        :return: the stress field for all lattice links at all lattice points of a given plane, accounting for boundary conditions; in plane dimensions (k,l,27,3,3)
        '''
        sigmaBd = copy.deepcopy(sigmaBd)
        for i in range(0, len(sigmaBd)):
            for j in range(0, len(sigmaBd[i])):
                for l in range(0, len(sigmaBd[i][j])):
                    for ii in range(0, len(sigmaBd[i][j][l])):
                        for jj in range(0, len(sigmaBd[i][j][l][ii])):
                            if (not np.isnan(sigmaBC[ii,jj])):
                                sigmaBd[i,j,l,ii,jj] = sigmaBC[ii,jj]
        return sigmaBd

    def computeSigmaBdWithoutExtrapolation(sigmaArg, ccArg, coordinateArg, coordinateValueArg):
        '''

        :param sigmaArg: the stress field in lattice dimensions (m,n,o,3,3)
        :param ccArg: array of lattice speeds, (27,3)
        :param coordinateArg: 'x', 'y', 'z' the coordinate direction identifying the plane
        :param coordinateValueArg: the index identifying the plane (either 0 or max in the respective direction)
        :return: the current stress field at the plane in plane dimensions (k,l,27,3,3)
        '''
        sigmaAtCoordinate = BC.selectAtCoordinate(sigmaArg,coordinateArg,coordinateValueArg)
        sigmaBd = np.zeros((len(sigmaArg),len(sigmaArg[0]),len(ccArg),3,3), dtype=np.double)

        for i in range(0, len(sigmaBd)):
            for j in range(0, len(sigmaBd[i])):
                for l in range(0, len(sigmaBd[i][j])):
                    sigmaBd[i,j,l] = sigmaAtCoordinate[i,j]
        return sigmaBd

    sigmaBd = applyTractionBoundaryConditions(computeSigmaBdWithoutExtrapolation(sigmaArg,ccArg,coordinateArg,coordinateValueArg), sigmaBC)
    return sigmaBd