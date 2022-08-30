import numpy as np
import math
import copy

nameOfSimulation = "Block3D"
pathToVTK = "/Users/alex/Desktop/"

lam = 1.0
mue = 1.0

rho0 = 1.0
P0 = np.zeros((3, 3))
j0 = np.zeros(3)

maxX = 10
maxY = maxX
maxZ = maxX
dx = 0.1  # spacing
xx = np.zeros((maxX, maxY, maxZ, 3), dtype = float)
for i in range(0, len(xx)):
    for j in range(0,len(xx[0])):
        for k in range(0, len(xx[0][0])):
            xx[i,j,k] = np.array([float(i) *dx, float(j) * dx, float(k) * dx], dtype=float)

cs = math.sqrt(mue/rho0)
dt = 1.0 / math.sqrt(3.0) * dx / cs
c = dx/dt
tau = 0.55*dt

#f = np.zeros((m, n, o, 27), dtype = float)
#fNew = np.zeros((m, n, o, 27), dtype = float)
#fEq = np.zeros((m, n, o, 27), dtype = float)

import Settings as SettingsModule
[cc, w] = SettingsModule.getLatticeVelocitiesWeights(c)

import Core
[f,j,P,u] = Core.intitialize(rho0, cs, cc, w, maxX, maxY, maxZ)
F = np.zeros((3), dtype=float)

import BoundaryConditions
import PostProcessing

tMax = 1.0
t = 0.0
k = int(0)
while(t <= tMax):
    fNew = np.zeros((maxX,maxY,maxZ,27), dtype=float)
    fNew.fill(np.nan)

    rho = Core.computeRho(f)
    S = Core.sourceTerm(dx,rho,lam,mue,F)
    j = Core.computeJ(f,S,cc,dt)
    P = Core.computeP(f,cc)
    u = Core.computeU(u, rho, j, dt)

    PostProcessing.writeVTKMaster(k,nameOfSimulation,pathToVTK,t,xx,u)

    fEq = Core.equilibriumDistribution(rho,j,P,cc,w,cs)
    psi = Core.sourceTermPsi(S,cc,w,cs)


    fColl = Core.collide(f,fEq,psi,dt,tau)
    fNew = Core.stream(fColl,cc,c)



    # apply BC at z=0
    sigmaBC = np.array([ [np.nan, np.nan, 0],
                         [np.nan, np.nan, 0],
                         [0.0, 0.0, 0.0]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew,fColl,u,rho, rho0, cs,cc,c,w,sigmaBC,dx,lam,mue,'z',0)

    # apply BC at z=max
    sigmaBC = np.array([[np.nan, np.nan, 0],
                        [np.nan, np.nan, 0],
                        [0.0, 0.0, 0.0]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBC, dx, lam, mue,
                                                             'z', maxZ-1)
    # apply BC at y=0
    sigmaBC = np.array([[np.nan, 0, np.nan],
                        [0, 0, 0],
                        [np.nan, 0, np.nan]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBC, dx, lam, mue,
                                                             'y', 0)

    # apply BC at y=max
    sigmaBC = np.array([[np.nan, 0, np.nan],
                        [0, 0, 0],
                        [np.nan, 0, np.nan]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBC, dx, lam, mue,
                                                             'y', maxY - 1)



    # apply BC at x=0
    sigmaBC = np.array([[0.000,   0.0, 0.0],
                        [0.0, np.nan, np.nan],
                        [0.0, np.nan, np.nan]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBC, dx, lam, mue,
                                                             'x', 0)

    # apply BC at x=max
    sigmaBC = np.array([[0.000,   0.0, 0.0],
                        [0.0,    np.nan, np.nan],
                        [0.0,    np.nan, np.nan]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBC, dx, lam, mue,
                                                             'x', maxX - 1)

    # apply BC at edge x = min, y = max
    sigmaBCXMax = np.array([[0.000, 0.0, 0.0],
                        [0.0, np.nan, np.nan],
                        [0.0, np.nan, np.nan]])
    sigmaBCYMAX = sigmaBC = np.array([[np.nan, 0, np.nan],
                        [0, 0, 0],
                        [np.nan, 0, np.nan]])
    fNew = BoundaryConditions.applyNeumannBoundaryConditionsAtEdge(fNew,fColl,u,rho,rho0,cs,cc,c,w,sigmaBCXMax,sigmaBCYMAX,dx,lam,mue,'x',0,'y',maxY-1)

    f = fNew

    k = k + 1
    print(k)
    t = t + dt


print("End")

