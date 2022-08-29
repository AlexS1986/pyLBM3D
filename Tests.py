import numpy as np
import math
import copy


lam = 1.0
mue = 1.0

rho0 = 1.0
P0 = np.zeros((3, 3))
j0 = np.zeros(3)


m = 2
n = m
o = m
dx = 0.1  # spacing

cs = math.sqrt(mue/rho0)
dt = 1.0 / math.sqrt(3.0) * dx / cs
c = dx/dt
tau = 0.55

f = np.zeros((m, n, o, 27), dtype = float)
f_new = np.zeros((m, n, o, 27), dtype = float)
f_eq = np.zeros((m, n, o, 27), dtype = float)

import Settings as SettingsModule
[cc, w] = SettingsModule.getLatticeVelocitiesWeights(c)
# print(f)


import PostProcessing

# test point plot
pointsTest = np.array([[2, 3, 4], [0, 0, 0]])
#PostProcessing.display(pointsTest)

import Core as Core

# test zeroth moment
ftest = f[0][0][0].sum()
zerothMoment = Core.zerothMoment(f)
zerothMomentTest = zerothMoment[0][0][0]


fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
firstMomentTest = Core.firstMoment(fTest, cc)
firstMomentTestValue = firstMomentTest[0][0][0]

secondMomentTest = Core.secondMoment(fTest, cc)
secondMomentTesta = secondMomentTest[0][0][0]

# test Gradient
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
dxTest = 1.0
rhoGradientTest = np.gradient(rhoTest, dxTest, dxTest, dxTest)
print(rhoTest.shape)
print(rhoGradientTest[0].shape)


# test sourceTerm
dxTest = 1.0
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
lamTest = 1.5
mueTest = 1.0
FTest = np.array([0.0, 0.0, 0.0])
sourceTermTest = Core.sourceTerm(dxTest, rhoTest, lamTest, mueTest, FTest)
print(sourceTermTest.shape)

# test equilibrium distribution
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
jTest = np.array( [ [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]]
                  ], dtype=float )
# PTest = np.array( [ [[ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] ,
#                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#                   [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] , [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
#                      [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] ,
#                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]], [[ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] , [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] ,
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] , [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]],
#                     [[ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] ,
#                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] , [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
#                      [ [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] ,
#                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
#                   ], dtype=float )
PTest = np.zeros((3, 3, 3, 3, 3), dtype=float)
print(PTest.shape)
feqOutTest = Core.equilibriumDistribution(rhoTest, jTest, PTest, cc, w, cs)
print(feqOutTest.shape)

#test stream
fTest = np.arange(m*n*o*27)
fTest = np.reshape(fTest,(m,n,o,27))
#fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
fStreamed = Core.stream(fTest, cc, c)
print(fStreamed[1][0][0][1])

# test collide
fTest = np.arange(m*n*o*27)
fTestEq = np.arange(m*n*o*27)
psiTest = np.arange(m*n*o*27)
dtTest = 0.1
tauTest = 0.55

fCollTest = Core.collide(fTest, fTestEq, psiTest, dtTest, tauTest)
print(fCollTest.shape)


# test SourceTermPsi
STest = np.array( [ [[ [0.0, 1.0, 0.0], [2.0, 0.0, 0.0] , [0.0, 0.0, 3.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]]
                  ], dtype=float )

sourceTermPsiTest = Core.sourceTermPsi(STest, cc, w, cs)
print(sourceTermPsiTest.shape)


# test computeJ
fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
STest = np.array( [ [[ [0.0, 1.0, 0.0], [2.0, 0.0, 0.0] , [0.0, 0.0, 3.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]]
                  ], dtype=float )
jTest = Core.computeJ(fTest, STest, cc, dtTest)
print(jTest.shape)


#test computeU
uOldTest = np.zeros((m, n, o, 3), dtype=float)
rhoTest = np.zeros((m, n, o), dtype=float)
rhoTest.fill(1.0)
jTest = np.zeros((m, n, o, 3), dtype=float)
jTest.fill(1.0)
dtTest = 0.1
uNewTest = Core.computeU(uOldTest, rhoTest, jTest, dtTest)
print(uNewTest.shape)


# test computeDivergence
uTest = np.zeros((m, n, o, 3), dtype=float)
print(uTest.shape)
divUTest = Core.computeDivergenceU(uTest, 0.1)
print(divUTest.shape)

import BoundaryConditions

# test selectAtCoordinate
fTest = np.zeros((m, n, o, 27), dtype = float)
print(fTest.shape)
fSelectedTest = BoundaryConditions.selectAtCoordinate(fTest, coordinateArg='z', coordinateValueArg=o-1)
print(fSelectedTest.shape)
fSelectedTest[0, 1, 0] = 989.0

# test select corner x=0, y=o, z=o
fXYZeq0 = BoundaryConditions.selectAtCoordinate(BoundaryConditions.selectAtCoordinate(BoundaryConditions.selectAtCoordinate(fTest, coordinateArg='x', coordinateValueArg=0)))
print(fXYZeq0.shape)
fXYZeq0[5] = 5


# test select interpolation neighbors
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.4 , 15.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
rhoNeighborsTest = BoundaryConditions.selectInterpolationNeighbors(rhoTest, cc, c, coordinateArg='z', coordinateValueArg=2)
print(rhoNeighborsTest.shape)


# test computeRhoBd
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.4 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 15.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
rhoBdTest = BoundaryConditions.extrapolateScalarToBd(rhoTest,cc, c, 'z', 2)
print(rhoBdTest)


# test computePBd
PTest = np.arange(3*3*3*3*3, dtype=float).reshape((3, 3, 3, 3, 3))
PBdTest = BoundaryConditions.extrapolateTensorToBd(PTest, cc, c, coordinateArg='z', coordinateValueArg=2)
print(PBdTest.shape)

# test getOppositeLatticeDirection
print(BoundaryConditions.getOppositeLatticeDirection(3))

# test dirichlet BC
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(np.nan)
fCollTest = np.arange(m*n*o*27, dtype = float).reshape((m,n,o,27))
rhoTest = np.zeros((m, n, o), dtype = float)
rhoTest.fill(1.0)
uBdTest = np.array([1,0,0])
fTestApplied = BoundaryConditions.applyDirichletBoundaryConditions(fTest, fCollTest, rhoTest, cs, cc, c, w, uBdTest)
print(fTestApplied.shape)


# test computeSigma
PTest = np.arange(m*n*o*3*3, dtype=float).reshape((m, n, o, 3, 3))
divUTest = np.arange(m*n*o, dtype=float).reshape((m, n, o))
laTest = 1.5
mueTest = 1.0
sigmaTest = Core.computeSigma(PTest, divUTest, laTest, mueTest)
print(sigmaTest.shape)


# test computePBd
sigmaBCTest = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, np.nan], [1.0, np.nan, np.nan]], dtype=float)
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(0.5)
uTest = np.arange(m*n*o*3, dtype=float).reshape((m,n,o,3))
PBdTest = BoundaryConditions.computePBd(sigmaBCTest,fTest,cc,c,uTest,0.1,1.5,1.0,'x',m-1)
print(PBdTest.shape)



# test Neumann BC
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(0.5)
fCollTest = np.arange(m*n*o*27, dtype = float).reshape((m,n,o,27))
rhoTest = np.zeros((m, n, o), dtype = float)
rhoTest.fill(1.0)
sigmaBCTest = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, np.nan], [1.0, np.nan, np.nan]], dtype=float)
uTest = np.arange(m*n*o*3, dtype=float).reshape((m,n,o,3))
uBdTest = np.array([1,0,0])

fTestApplied = BoundaryConditions.applyNeumannBoundaryConditions(fTest,fCollTest,uTest,rhoTest,cs,cc,c,w,sigmaBCTest,0.1,1.5,1.9,'x',m-1)
#fTestApplied = applyDirichletBoundaryConditions(fTest, fCollTest, rhoTest, cs, cc, c, w, uBdTest)
print(fTestApplied.shape)


xTest = np.zeros((m, n, o, 3), dtype = float)
dx = 0.1
points = list()

for i in range(0, len(xTest)):
    for j in range(0,len(xTest[0])):
        for k in range(0, len(xTest[0][0])):
            xTest[i,j,k] = np.array([float(i) *dx, float(j) * dx, float(k) * dx], dtype=float)
            ppData = dict()
            ppData["u"] = [1.0,0.0,0.0]
            points.append(PostProcessing.Point(float(i) *dx, float(j) * dx, float(k) * dx, ppData))

PostProcessing.writeVTK(1,"test","/Users/alex/Desktop/",points,0.1)



print("Test end.")