import numpy as np
import math
import Settings as SettingsModule
import Experimental as Ex
import PostProcessing
import Core
import Util

nameOfSimulation = "Block3D"
pathToVTK = "./vtk/"

lam = 1.0
mue = 1.0

rho0 = 1.0
P0 = np.zeros((3, 3))
j0 = np.zeros(3)

ax = 1.0
maxX = 20
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
tau = 0.55 * dt

#print(np.__version__)

#f = np.zeros((m, n, o, 27), dtype = np.double)
#fNew = np.zeros((m, n, o, 27), dtype = np.double)
#fEq = np.zeros((m, n, o, 27), dtype = np.double)


[cc, w] = SettingsModule.getLatticeVelocitiesWeights(c)


[f, j, sigma, u] = Ex.intitialize(rho0, cs, cc, w, maxX, maxY, maxZ, lam, mue)
b = np.zeros((maxX, maxY, maxZ, 3), dtype=np.double)


tMax = 2.0
t = 0.0
k = int(0)

while t <= tMax:
    fNew = np.zeros((maxX, maxY, maxZ, 27), dtype=np.double)
    fNew.fill(np.nan)

    rho = Core.computeRho(f)
    jOld = j
    j = Ex.j(Core.firstMoment(f, cc), dt, Ex.firstSource(b, rho0))
    sigma = Ex.sigma(Core.secondMoment(f, cc), dt, Ex.secondSource(Util.dJyDy(j, dx), lam, mue, rho0))
    u = Ex.computeU(u, rho0, j, jOld, dt)

    PostProcessing.writeVTKMaster(k, nameOfSimulation, pathToVTK, t, xx, u, sigma)

    fEq = Ex.equilibriumDistribution(rho, j, sigma, cc, w, cs, lam, mue, rho0)
    psi = Ex.sourceTermPsi(b, rho0, Util.dJyDy(j, dx), cc, w, cs, mue, lam)

    fColl = Core.collide(f, fEq, psi, dt, tau)
    #print(fNew)
    fNew = Core.stream(fColl, cc, c)
    #print(fNew)

    sigmaXX = 0.001
    uBdDesired = np.array([0.0, 0.0, 0.0], dtype=np.double)

    sigmaBCXMax = np.array([[sigmaXX, 0.0, 0.0],
                            [0.0, np.nan, np.nan],
                            [0.0, np.nan, np.nan]])
    #sigmaBCXMin = np.array([[np.nan, 0.0, 0.0],
    #                        [0.0, np.nan, np.nan],
    #                        [0.0, np.nan, np.nan]])
    sigmaBCXMin = np.array([[np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan]])

    sigmaBCYMin = np.array([[np.nan, 0, np.nan],
                        [0, 0, 0],
                        [np.nan, 0, np.nan]])
    sigmaBCYMax = sigmaBCYMin

    sigmaBCZMin = sigmaBC = np.array([[np.nan, np.nan, 0],
                        [np.nan, np.nan, 0],
                        [0.0, 0.0, 0.0]])
    sigmaBCZMax = sigmaBCZMin


    #edges
    sigmaBCEdgeXY = np.array([[sigmaXX, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, np.nan]])

    sigmaBCEdgeXY_XMin = np.array([[np.nan, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, np.nan]])

    sigmaBCEdgeYZ = np.array([[np.nan, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0]])

    sigmaBCEdgeXZ = np.array([[sigmaXX, 0.0, 0.0],
                              [0.0, np.nan, 0.0],
                              [0.0, 0.0, 0.0]])

    sigmaBCEdgeXZ_XMin = np.array([[np.nan, 0.0, 0.0],
                              [0.0, np.nan, 0.0],
                              [0.0, 0.0, 0.0]])

    # corner
    sigmaBCCorner = np.array([[sigmaXX, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])

    sigmaBCCorner_XMin = np.array([[np.nan, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])

    # ### test
    #
    # sigmaBCXMin = np.array([[0.001, 0.0, 0.0],
    #                         [0.0, 0.0, 0.0],
    #                         [0.0, 0.0, 0.0]])
    # sigmaBCXMax = sigmaBCXMin
    #
    # sigmaBCYMin = np.array([[0, 0, 0],
    #                     [0, 0, 0],
    #                     [0, 0, 0]])
    # sigmaBCYMax = sigmaBCYMin
    #
    # sigmaBCZMin = sigmaBC = np.array([[0, 0, 0],
    #                     [0, 0, 0],
    #                     [0.0, 0.0, 0.0]])
    # sigmaBCZMax = sigmaBCZMin


    # apply BC at z=0
    fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho, cs, cc, w, sigmaBCZMin, sigma, 'z', 0)
    #fNew = Ex.applyNeumannBoundaryConditions(fArg, fCollArg, rhoArg, csArg, ccArg, wArg, sigmaBdArg, sigmaArg, coordinateArg='x', coordinateValueArg=0)

    # apply BC at z=max

    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCZMax, sigma, dx, lam, mue,
    #                                                         'z', maxZ-1)
    fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho, cs, cc, w, sigmaBCZMax, sigma,
                                             'z', maxZ - 1)
    # apply BC at y=0
    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCYMin, sigma, dx, lam, mue,
    #                                         'y', 0)
    fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho,  cs, cc, w, sigmaBCYMin, sigma,
                                                             'y', 0)

    # apply BC at y=max
    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCYMax, sigma, dx, lam, mue,
    #                                         'y', maxY - 1)
    fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho, cs, cc, w, sigmaBCYMax, sigma,
                                                             'y', maxY - 1)

    # apply BC at x=0
    #uBdOld = u
    jBd = np.array([0, 0, 0])
    fNew = Ex.applyDirichletBoundaryConditions(fNew, fColl, rho, cs, cc, c, w, jBd, 'x', 0)

    #fNew = Ex.applyDirichletBoundaryConditions(fNew, fColl, rho0, cs, cc, w, uBdDesired, u, j, dt, 'x', 0)

    # apply BC at x=0
    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho, cs, cc, w, sigmaBCXMin, sigma,
    #                                                         'x', 0)
    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCXMin, sigma, dx, lam, mue,
    #                                         'x', 0)

    # apply BC at x=max
    #fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCXMax, sigma, dx, lam, mue,
    #                                                         'x', maxX - 1)
    fNew = Ex.applyNeumannBoundaryConditions(fNew, fColl, rho, cs, cc, w, sigmaBCXMax, sigma,
                                             'x', maxX - 1)

    ## Edges ##



    # apply BC at edge x = min, y = min

    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXY,
    #                                                               sigmaBCEdgeXY, sigma, dx, lam, mue, 'x', 0, 'y', 0)
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, sigmaBCEdgeXY,
    #                                               sigmaBCEdgeXY, sigma, 'x', 0, 'y', 0)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXY_XMin, sigma, 'x', 0, 'y', 0)


    # apply BC at edge x = min, y = max
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew,fColl,u,rho,rho0,cs,cc,c,w,sigmaBCEdgeXY,sigmaBCEdgeXY, sigma, dx, lam, mue, 'x', 0, 'y', maxY-1)
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, sigmaBCEdgeXY,
    #                                               sigmaBCEdgeXY, sigma,  'x', 0, 'y', maxY - 1)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXY_XMin, sigma, 'x', 0, 'y', maxY - 1)

    # apply BC at edge x = min, z = min
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXZ,
    #                                                               sigmaBCEdgeXZ, sigma, dx, lam, mue, 'x', 0, 'z', 0)
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeXZ,
    #                                               sigmaBCEdgeXZ, sigma, 'x', 0, 'z', 0)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXZ_XMin, sigma, 'x', 0, 'z', 0)

    # apply BC at edge x = min, z = max
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXZ,
    #                                                               sigmaBCEdgeXZ, sigma, dx, lam, mue, 'x', 0, 'z', maxZ-1)
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeXZ,
    #                                               sigmaBCEdgeXZ, sigma,  'x', 0, 'z', maxZ - 1)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXZ_XMin, sigma, 'x', 0, 'z', maxZ -1)






    # apply BC at edge x = max, y = min
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXY,
    #                                                               sigmaBCEdgeXY, sigma, dx, lam, mue, 'x', maxX-1, 'y', 0)
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, rho,  cs, cc, w, sigmaBCEdgeXY,
                                                   sigmaBCEdgeXY, sigma, 'x', maxX - 1, 'y', 0)

    # apply BC at edge x = max, y = max
    #sigmaBCYMax = sigmaBCYMin
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew,fColl,u,rho,rho0,cs,cc,c,w,sigmaBCEdgeXY,
    #                                                               sigmaBCEdgeXY, sigma, dx,lam,mue,'x',maxX-1,'y',maxY-1)
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, rho,  cs, cc,  w, sigmaBCEdgeXY,
                                                   sigmaBCEdgeXY, sigma,  'x', maxX - 1, 'y', maxY - 1)

    # apply BC at edge x = max, z = min
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXZ,
    #                                                               sigmaBCEdgeXZ, sigma, dx, lam, mue, 'x', maxX-1, 'z', 0)
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, rho,  cs, cc,  w, sigmaBCEdgeXZ,
                                                                   sigmaBCEdgeXZ, sigma,  'x', maxX-1, 'z', 0)

    # apply BC at edge x = max, z = max
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeXZ,
    #                                                               sigmaBCEdgeXZ, sigma, dx, lam, mue, 'x', maxX-1, 'z', maxZ-1)
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho, cs, cc,  w, sigmaBCEdgeXZ,
                                                                   sigmaBCEdgeXZ, sigma, 'x', maxX-1, 'z', maxZ-1)




    # apply BC at edge y = min, z = min
    #fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl, u, rho, rho0, cs, cc, c, w, sigmaBCEdgeYZ ,
    #                                                               sigmaBCEdgeYZ, sigma, dx, lam, mue, 'y', 0, 'z', 0)
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeYZ,
                                                   sigmaBCEdgeYZ, sigma,  'y', 0, 'z', 0)

    # apply BC at edge y = min, z = max
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeYZ ,
                                                                   sigmaBCEdgeYZ, sigma,  'y', 0, 'z', maxZ-1)

    # apply BC at edge y = max, z = min
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeYZ ,
                                                                   sigmaBCEdgeYZ, sigma,  'y', maxY-1, 'z', 0)

    # apply BC at edge y = max, z = max
    fNew = Ex.applyNeumannBoundaryConditionsAtEdge(fNew, fColl,  rho,  cs, cc,  w, sigmaBCEdgeYZ ,
                                                                   sigmaBCEdgeYZ, sigma, 'y', maxY-1, 'z', maxZ-1)

    ### Dirichlet B.C.

    # apply BC at edge x = min, y = min
    fNew = Ex.applyDirichletBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, c, w, jBd, 'x', 0, 'y',
                                                     0)

    # apply BC at edge x = min, y = max
    fNew = Ex.applyDirichletBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, c, w, jBd, 'x', 0, 'y',
                                                     maxY - 1)

    # apply BC at edge x = min, z = min
    fNew = Ex.applyDirichletBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, c, w, jBd, 'x', 0, 'y',
                                                     0)

    # apply BC at edge x = min, z = max
    fNew = Ex.applyDirichletBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, c, w, jBd, 'x', 0, 'y',
                                                     maxZ - 1)

    # apply BC at edge x = min, y = min
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXY_XMin, sigma, 'x', 0, 'y', 0)

    # apply BC at edge x = min, y = max
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXY_XMin, sigma, 'x', 0, 'y', maxY - 1)

    # apply BC at edge x = min, z = min
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXZ_XMin, sigma, 'x', 0, 'z', 0)

    # apply BC at edge x = min, z = max
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtEdge(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0, u, j, dt,
    #                                                        sigmaBCEdgeXZ_XMin, sigma, 'x', 0, 'z', maxZ -1)


    ## Corners ##

    #xmin, ymin, zmin
    #fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho, cs, cc, w,sigmaBCCorner,sigmaBCCorner,sigmaBCCorner, sigma, 0,0,0)

    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0,
    #                                                          u, j, dt, sigmaBCCorner_XMin, sigmaBCCorner_XMin, sigma,
    #                                                          0, 0, 0)

    # xmax, ymin, zmin
    fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho,  cs, cc,  w,
                                                                     sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
                                                                      maxX-1, 0, 0)
    # xmax, ymax, zmin
    fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho,  cs, cc,  w,
                                                                     sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
                                                                      maxX - 1, maxY-1, 0)
    # xmin, ymax, zmin
    #fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho,  cs, cc,  w,
    #                                                                 sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
    #                                                                  0, maxY-1, 0)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0,
    #                                                          u, j, dt, sigmaBCCorner_XMin, sigmaBCCorner_XMin, sigma,
    #                                                          0, maxY-1, 0)


    #xmin, ymin, zmax
    #fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew,fColl,rho,cs,cc,w,sigmaBCCorner,sigmaBCCorner,sigmaBCCorner, sigma, 0,0,maxZ-1)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0,
    #                                                          u, j, dt, sigmaBCCorner_XMin, sigmaBCCorner_XMin, sigma,
    #                                                          0, 0, maxZ-1)


    # xmax, ymin, zmax
    fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho,  cs, cc,  w,
                                                                     sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
                                                                      maxX-1, 0, maxZ-1)
    # xmax, ymax, zmax
    fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl, rho,  cs, cc,  w,
                                                                     sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
                                                                      maxX - 1, maxY-1, maxZ-1)
    # xmin, ymax, zmax
    #fNew = Ex.applyNeumannBoundaryConditionsAtCorner(fNew, fColl,  rho,  cs, cc,  w,
    #                                                                 sigmaBCCorner, sigmaBCCorner, sigmaBCCorner, sigma,
    #                                                                  0, maxY-1, maxZ-1)
    #fNew = Ex.applyDirichletNeumannBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc, w, uBdDesired, rho0,
    #                                                          u, j, dt, sigmaBCCorner_XMin, sigmaBCCorner_XMin, sigma,
    #                                                          0, maxY-1, maxZ-1)

    # xmin, ymin, zmin
    fNew = Ex.applyDirichletBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc,  w, jBd,  0,
                                                       0, 0)

    # xmin, ymax, zmin
    fNew = Ex.applyDirichletBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc,  w, jBd,  0,
                                                       maxY - 1, 0)

    # xmin, ymin, zmax
    fNew = Ex.applyDirichletBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc,  w, jBd,  0,
                                                       0, maxZ - 1)

    # xmin, ymax, zmax
    fNew = Ex.applyDirichletBoundaryConditionsAtCorner(fNew, fColl, rho, cs, cc,  w, jBd,  0,
                                                       maxY - 1, maxZ - 1)


    f = fNew

    k = k + 1
    print(k)
    t = t + dt


print("End")

