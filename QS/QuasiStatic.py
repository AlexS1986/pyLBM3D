import numpy as np

def intitialize(rho0Arg,  wArg, mArg, nArg, oArg):
    '''
    returns the initial distribution functions (as equilibrium distributions), moments and displacement field
    :param rho0Arg: the initial density, a scalar
    :param wArg: weights of lattice directions (27)
    :param mArg: dimension of lattice in x direction
    :param nArg: dimension of lattice in y direction
    :param oArg: dimension of lattice in z direction
    :return: [fOut, j0, sigma0, u0]
    '''
    sigma0 = np.zeros((mArg, nArg, oArg, 3, 3), dtype=np.double)
    j0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    u0 = np.zeros((mArg, nArg, oArg, 3), dtype=np.double)
    rho = np.zeros((mArg, nArg, oArg), dtype=np.double)
    rho.fill(rho0Arg)
    fOut = equilibriumDistribution(rho, wArg=wArg)
    return [fOut, j0, sigma0, u0]

def gi(gArg, ccArg, wArg, rhoArg, csArg, vArg):
    gi = np.zeros((len(gArg), len(gArg[0]), len(gArg[0][0]), len(ccArg)), dtype=np.double)
    for i in range(0, len(gArg)):
        for j in range(0, len(gArg[0])):
            for k in range(0, len(gArg[0][0])):
                for l in range(0,len(ccArg)):
                    bracket = np.einsum('a,a',ccArg[l],vArg[i,j,k]) * ccArg[l] - 1.0 / csArg ** 2 * vArg[i,j,k]
                    gi[i,j,k,l] = wArg[l] * rhoArg[i,j,k] / csArg ** 2 * np.einsum('a,a',ccArg[l],gArg[i,j,k]) # page 4
                    + wArg[l] * rhoArg[i,j,k] / 2.0 / ( csArg ** 4) * np.einsum('a,a',bracket, gArg[i,j,k])
    return gi               
    

def g(rhoArg,divSigmaArg):
    g = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(rhoArg)):
        for j in range(0, len(rhoArg[0])):
            for k in range(0, len(rhoArg[0][0])):
                g[i,j,k] = 1.0/rhoArg[i,j,k] * divSigmaArg[i,j,k]
    return g

def firstSource(rhoArg, divSigmaArg):
    return g(rhoArg, divSigmaArg)


def divOfSigma(sigmaArg, dxArg):
    divOfSigma = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0]),3), dtype=np.double)

    S11 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S12 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S13 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)

    S21 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S22 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S23 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)

    S31 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S32 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    S33 = np.zeros((len(sigmaArg), len(sigmaArg[0]), len(sigmaArg[0][0])), dtype=np.double)
    
    for i in range(0, len(sigmaArg)):
        for j in range(0, len(sigmaArg[0])):
            for k in range(0, len(sigmaArg[0][0])):
                S11[i,j,k] = sigmaArg[i,j,k,0,0]
                S12[i,j,k] = sigmaArg[i,j,k,0,1]
                S13[i,j,k] = sigmaArg[i,j,k,0,2]

                S21[i,j,k] = sigmaArg[i,j,k,1,0]
                S22[i,j,k] = sigmaArg[i,j,k,1,1]
                S23[i,j,k] = sigmaArg[i,j,k,1,2]

                S31[i,j,k] = sigmaArg[i,j,k,2,0]
                S32[i,j,k] = sigmaArg[i,j,k,2,1]
                S33[i,j,k] = sigmaArg[i,j,k,2,2]
        

    gradientS11 = np.gradient(S11, dxArg, dxArg, dxArg, edge_order=2)
    gradientS12 = np.gradient(S12, dxArg, dxArg, dxArg, edge_order=2)
    gradientS13 = np.gradient(S13, dxArg, dxArg, dxArg, edge_order=2)

    gradientS21 = np.gradient(S21, dxArg, dxArg, dxArg, edge_order=2)
    gradientS22 = np.gradient(S22, dxArg, dxArg, dxArg, edge_order=2)
    gradientS23 = np.gradient(S23, dxArg, dxArg, dxArg, edge_order=2)

    gradientS31 = np.gradient(S31, dxArg, dxArg, dxArg, edge_order=2)
    gradientS32 = np.gradient(S32, dxArg, dxArg, dxArg, edge_order=2)
    gradientS33 = np.gradient(S33, dxArg, dxArg, dxArg, edge_order=2)


    divOfSigma[:,:,:,0] = gradientS11[0] + gradientS12[1] + gradientS13[2]
    divOfSigma[:,:,:,1] = gradientS21[0] + gradientS22[1] + gradientS23[2]
    divOfSigma[:,:,:,2] = gradientS31[0] + gradientS32[1] + gradientS33[2]

    return divOfSigma


def collide(fArg, feqArg, giArg, dtArg, omegaArg):
    fCollOut = fArg - dtArg*omegaArg * (fArg - feqArg) - dtArg * (1.0 - omegaArg / 2.0) * giArg
    return fCollOut





def equilibriumDistribution(rhoArg,  wArg):
    feqOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 27), dtype=np.double)
    for i in range(0, len(feqOut)):
        for j in range(0, len(feqOut[0])):
            for k in range(0, len(feqOut[0][0])):
                for l in range(0, len(feqOut[0][0][0])):
                    feqOut[i][j][k][l] = wArg[l] * rhoArg[i][j][k]
    return feqOut

def linearizedStrain(gradUArg):
    epsOut = np.zeros((len(gradUArg), len(gradUArg[0]), len(gradUArg[0][0]), 3, 3), dtype=np.double)
    for i in range(0, len(epsOut)):
        for j in range(0, len(epsOut[0])):
            for k in range(0, len(epsOut[0][0])):
                epsOut[i,j,k] = 0.5 * gradUArg[i,j,k].transpose() + 0.5 * gradUArg[i,j,k]
    return epsOut

def sigmaFromDisplacement(gradUArg,laArg,mueArg):
    sigmaOut = np.zeros((len(gradUArg), len(gradUArg[0]), len(gradUArg[0][0]), 3, 3), dtype=np.double)
    eps = linearizedStrain(gradUArg)
    I = np.identity(3, dtype=np.double)
    for i in range(0, len(sigmaOut)):
        for j in range(0, len(sigmaOut[0])):
            for k in range(0, len(sigmaOut[0][0])):
                sigmaOut[i,j,k] = laArg * I * eps[i,j,k].trace() + 2.0 * mueArg * eps[i,j,k]
    return sigmaOut

    
def v(rhoArg, jArg):
    vOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 3), dtype=np.double)
    for i in range(0, len(rhoArg)):
        for j in range(0, len(rhoArg[0])):
            for k in range(0, len(rhoArg[0][0])):
                vOut[i,j,k] = 1.0 / rhoArg[i,j,k] * jArg[i,j,k]
    return vOut


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
                if(i == 0 or j == 0 or k == 0 or i == len(uOldArg) or j == len(uOldArg[0]) or k == len(uOldArg[0][0])):
                    uNew[i][j][k] = uOldArg[i][j][k]
                else:
                    uNew[i][j][k] = uOldArg[i][j][k] + (jArg[i][j][k] + jOldArg[i][j][k]) / rho0Arg / 2.0 * dtArg
    return uNew





    
