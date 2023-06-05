import numpy as np
import math
import Settings as SettingsModule
import Experimental as Ex
import PostProcessing
import Core
import Util

import QS.QuasiStatic as QS
import QS.QuasiStaticBC as QSBC

nameOfSimulation = "Block3D"
pathToVTK = "vtk/"

lam = 1.0
mue = 1.0

rho0 = 1.0
P0 = np.zeros((3, 3))
j0 = np.zeros(3)

# setting up lattice point positions
ax = 1.0
maxX = 10
maxY = maxX
maxZ = maxX
dx = ax/maxX  # spacing
xx = np.zeros((maxX, maxY, maxZ, 3), dtype=np.double)
for i in range(0, len(xx)):
    for j in range(0,len(xx[0])):
        for k in range(0, len(xx[0][0])):
            xx[i, j, k] = np.array([np.double(i) * dx, np.double(j) * dx, np.double(k) * dx], dtype=np.double)

cs = math.sqrt(mue/rho0)
dt = 1.0 / math.sqrt(3.0) * dx / cs
c = dx / dt

nu = lam / (2.0 * (lam + mue))
omega = 1.0 # one should also work, 2.0 * cs ** 2 / ( cs ** 2 + 2 * nu) # TODO do you use this for solids als well (page 4)
tau = 1.0 / omega
#tau = 0.55 * dt


[cc, w] = SettingsModule.getLatticeVelocitiesWeights(c)

[f, j, sigma, u] = QS.intitialize(rho0, w, maxX, maxY, maxZ)
divSigma = QS.divOfSigma(sigma,dx)
rho = Core.computeRho(f)
#b = np.zeros((maxX, maxY, maxZ, 3), dtype=np.double) # TODO not needed


tMax = 2.0
t = 0.0
k = int(0)



outputFile = None
pointIndices = [[maxX-1, maxY-1, maxZ-1], [maxX-1, maxY-1, maxZ-1], [0,0,0]]

while t <= tMax:
    fNew = np.zeros((maxX, maxY, maxZ, 27), dtype=np.double)
    fNew.fill(np.nan)

    # Lattice forces
    g = QS.g(rho, divSigma)
    v = QS.v(rho,j)
    gi = QS.gi(g,cc,w,rho,cs,v) # TODO this is cs = 1/sqrt(3)

    # BC ####################################
    def uBdFromCoordinates(coordinates):
        clocal = 0.001  # TODO maybe ramping
        nu = lam / (2.0 * (lam + mue))
        return np.array([ clocal * coordinates[0], - nu * clocal * coordinates[1], - nu * clocal * coordinates[2] ])

    # TODO for fixed BC use bounce back, halfway? 
    #xmin
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'x',0)

    #xmax
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'x',maxX-1)

    #ymin
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'y',0)

    #ymax
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'y',maxY-1)

    #zmin
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'z',0)

    #zmax
    [f,u] = QSBC.applyDirichletBoundaryConditions(f,dx,dt,rho0,w,u,uBdFromCoordinates,'z',maxZ-1)
    
    # End BC #################################

    # Postprocessing ################
    PostProcessing.writeVTKMaster(k, nameOfSimulation, pathToVTK, t, xx, u, sigma)
    uOut = []
    for indices in pointIndices:
        uOut.append(u[indices[0], indices[1], indices[2]])

    outputFile = PostProcessing.writeToFile(outputFile,"./DirichletQS.dis",uOut,t)
    ######################

    # collision and streaming
    fEq = QS.equilibriumDistribution(rho,w)

    fColl = QS.collide(f,fEq,gi,dt,omega)
    fNew = Core.stream(fColl, cc, c)
    f = fNew

    # compute displacement
    jOld = j
    j = Ex.j(Core.firstMoment(f, cc), dt, g)
     # compute rho
    rho = Core.computeRho(f)
    u = QS.computeU(u, rho0, j, jOld, dt) # TODO rho0 more stable or use rho?

   
    # compute strain, stress, stress divergence
    gradU = Util.computeGradientU(u,dx)
    sigma = QS.sigmaFromDisplacement(gradU,lam,mue)
    divSigma = QS.divOfSigma(sigma,dx)

    k = k + 1
    print(k)
    t = t + dt

if outputFile is not None:
    outputFile.close()
print("End")

