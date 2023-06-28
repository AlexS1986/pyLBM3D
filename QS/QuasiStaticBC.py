import os, sys

parent = os.path.abspath('.')
sys.path.insert(1, parent)


import numpy as np
import copy 
import BoundaryConditions as BC
import Util as util
import QS.QuasiStatic as QSt 


def applyDirichletBoundaryConditions(fArg, dxArg, dtArg, rho0Arg, rhoArg, wArg,   uArg, uBdFromCoordinatesFunction, visited,
                                     coordinateArg='x', coordinateValueArg=0, omegaBCArg=1.0):
    def densityLinearTheory(gradUArg,rho0Arg):
        
        eps = QSt.linearizedStrain(gradUArg)
        trEps = util.trace(eps)
        rhoOut = np.zeros((len(eps), len(eps[0]), len(eps[0][0])), dtype=np.double)
        for i in range(0, len(rhoOut)):
            for j in range(0, len(rhoOut[0])):
                for k in range(0, len(rhoOut[0][0])):
                    rhoOut[i,j,k] = 1.0 / (1.0 + trEps[i,j,k]) * rho0Arg
        return rhoOut


    def densityFromDisplacementField(uArg, dxArg, rho0Arg, densityFromGradientUFunction=densityLinearTheory):
        gradU = util.computeGradientU(uArg,dxArg)
        rhoOut = densityFromGradientUFunction(gradU,rho0Arg)
        return rhoOut

    def computeCoordinatesFromIndices(dxArg,coordinateArg, coordinateValueArg, iArg,jArg):
        if(coordinateArg == "x"):
            xOut = coordinateValueArg * dxArg
            yOut = iArg * dxArg
            zOut = jArg * dxArg
        elif(coordinateArg == "y"):
            xOut = iArg * dxArg
            yOut = coordinateValueArg * dxArg
            zOut = jArg * dxArg
        elif(coordinateArg == "z"):
            xOut = iArg * dxArg
            yOut = jArg * dxArg
            zOut = coordinateValueArg * dxArg
        return [xOut, yOut, zOut]
    
    

    # set displacement field to prescribed value
    uOut = copy.deepcopy(uArg)
    uAtBC = BC.selectAtCoordinate(uOut, coordinateArg, coordinateValueArg)
    for i in range(0, len(uAtBC)):
        for j in range(0, len(uAtBC[0])):
            uAtBC[i,j] = uBdFromCoordinatesFunction(computeCoordinatesFromIndices(dxArg,coordinateArg,coordinateValueArg,i,j))

    # compute new density at boundary lattice points # TODO done differently, maybe dont do this step and use old rho
    # rho = densityFromDisplacementField(uOut,dxArg,rho0Arg)

    # compute equilibrium distribution at lattice points
    rho=rhoArg
    fEq = QSt.equilibriumDistribution(rho,wArg)

    # compute unknown distribution functions by relaxing
    fOut = copy.deepcopy(fArg)
    fAtBC = BC.selectAtCoordinate(fOut, coordinateArg, coordinateValueArg)
    fEqAtBC = BC.selectAtCoordinate(fEq, coordinateArg, coordinateValueArg)

    visitedOut = copy.deepcopy(visited)
    visitedAtBc = BC.selectAtCoordinate(visitedOut, coordinateArg, coordinateValueArg)
    for i in range(0, len(fAtBC)):
        for j in range(0, len(fAtBC[0])):
            if(not visitedAtBc[i,j]): # lattice point has not been visited yet
                # Collide without source -> double collision 
                fAtBC[i,j] = fAtBC[i,j] - dtArg * omegaBCArg * (fAtBC[i,j] - fEqAtBC[i,j])
                visitedAtBc[i,j] = True
            
    return [fOut, uOut, visitedOut]