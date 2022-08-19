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

cc = np.array([[0.0, 0.0, 0.0],  [c, 0.0, 0.0], [-c, 0, 0], [0, c, 0], [0, -c, 0], [0, 0, c], [0, 0, -c],  # 0  - 6
               [c, c, 0], [-c, -c, 0], [c, 0, c], [-c, 0, -c], [0, c, c], [0, -c, -c],  # 7-12
               [c, -c, 0], [-c, c, 0], [c, 0, -c], [-c, 0, c], [0, c, -c], [0, -c, c],  # 13-18
               [c, c, c], [-c, -c, -c], [c, c, -c], [-c, -c, c], [c, -c, c], [-c, c, -c], [-c, c, c], [c, -c, -c]  # 19 - 26
               ], dtype = float)
w = np.array([8.0/27.0,  # 0
              2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0, 2.0/27.0,  # 1-6
              1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0, 1.0/54.0,  # 7  - 18
              1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0, 1.0/216.0], dtype = float)  # 19 - 26


# print(f)


def display(points):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(111, projection='3d')
    for point in points:
        ax.scatter(point[0], point[1], point[2], c='r')  # plot the point (2,3,4) on the figure
    plt.show()
    return True

# test point plot
pointsTest = np.array([[2, 3, 4], [0, 0, 0]])
#display(pointsTest)


def zerothMoment(fArg):
    zerothMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0])))
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0,len(fArg[0][0])):
                zerothMomentOut[i][j][k] = fArg[i][j][k].sum()
    return zerothMomentOut

# test zeroth moment
ftest = f[0][0][0].sum()
zerothMoment = zerothMoment(f)
zerothMomentTest = zerothMoment[0][0][0]


def firstMoment(fArg, ccArg):
    firstMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 3))
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0, len(fArg[0][0])):
                firstMomentOut[i][j][k] = np.zeros(3, dtype=float)
                for l in range(0, len(ccArg)):
                    firstMomentOut[i][j][k] = firstMomentOut[i][j][k] + fArg[i][j][k][l] * ccArg[l]
    return firstMomentOut


fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
firstMomentTest = firstMoment(fTest, cc)
firstMomentTestValue = firstMomentTest[0][0][0]

def secondMoment(fArg, ccArg):
    secondMomentOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 3, 3), dtype=float)
    for i in range(0, len(fArg)):
        for j in range(0, len(fArg[0])):
            for k in range(0, len(fArg[0][0])):
                secondMomentOut[i][j][k] = np.zeros((3, 3), dtype=float)
                for l in range(0, len(ccArg)):
                    secondMomentOut[i][j][k] = secondMomentOut[i][j][k] + fArg[i][j][k][l] * np.outer(ccArg[l], ccArg[l].transpose())
    return secondMomentOut


secondMomentTest = secondMoment(fTest, cc)
secondMomentTesta = secondMomentTest[0][0][0]


def sourceTerm(dxArg, rhoArg, lamArg, mueArg, FArg):
    sourceTermOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 3), dtype=float)
    gradientRho = np.gradient(rhoArg, dxArg, dxArg, dxArg)
    for i in range(0, len(rhoArg)):
        for j in range(0, len(rhoArg[0])):
            for k in range(0, len(rhoArg[0][0])):
                sourceTermOut[i][j][k] = FArg + 1.0/rhoArg[i][j][k]*(mueArg-lamArg)*np.array([gradientRho[0][i][j][k], gradientRho[1][i][j][k], gradientRho[2][i][j][k]])

    return sourceTermOut


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
sourceTermTest = sourceTerm(dxTest, rhoTest, lamTest, mueTest, FTest)
print(sourceTermTest.shape)


def equilibriumDistribution(rhoArg, jArg, PArg, ccArg, wArg, csArg):
    feqOut = np.zeros((len(rhoArg), len(rhoArg[0]), len(rhoArg[0][0]), 27), dtype=float)
    for i in range(0, len(feqOut)):
        for j in range(0, len(feqOut[0])):
            for k in range(0, len(feqOut[0][0])):
                for l in range(0, len(feqOut[0][0][0])):
                    tmp2 = np.tensordot((PArg[i][j][k] - rhoArg[i][j][k] * cs ** 2 * np.identity(3, dtype=float)), (np.outer(ccArg[l], ccArg[l].transpose()) - cs ** 2 * np.identity(3, dtype=float)), axes=2)
                    tmp1 = np.tensordot(ccArg[l], jArg[i][j][k], axes=1)
                    feqOut[i][j][k][l] = wArg[l] * (rhoArg[i][j][k] + 1.0/(csArg ** 2) * tmp1 + 1.0 / (2.0 * csArg ** 4) * tmp2)
    return feqOut


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
feqOutTest = equilibriumDistribution(rhoTest, jTest, PTest, cc, w, cs)
print(feqOutTest.shape)

def checkIfIndicesInArrayBounds(iArg, jArg, kArg, arrayArg):
    return iArg < len(arrayArg)  and iArg >= 0 and jArg < len(arrayArg[0]) and jArg >= 0 and kArg < len(arrayArg[0][0]) and kArg >= 0


def stream(fArg, ccArg , cArg):
   fOut =  np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), 27), dtype=float)

   for i in range(0, len(fOut)):   # f0
       for j in range(0, len(fOut[0])):
           for k in range(0, len(fOut[0][0])):
               for l in range(0, len(fOut[0][0][0])):
                   cL = 1.0 / cArg * ccArg[l]
                   indicesToStreamTo = [int(i + cL[0]), int(j + cL[1]), int(k + cL[2])]
                   if checkIfIndicesInArrayBounds(indicesToStreamTo[0], indicesToStreamTo[1], indicesToStreamTo[2], fOut):
                   #if indicesToStreamTo[0] < len(fOut) and indicesToStreamTo[1] < len(fOut[0]) and indicesToStreamTo[2] < len(fOut[0][0]):
                       fOut[indicesToStreamTo[0]][indicesToStreamTo[1]][indicesToStreamTo[2]][l] = fArg[i][j][k][l]
   return fOut

#test stream
fTest = np.arange(m*n*o*27)
fTest = np.reshape(fTest,(m,n,o,27))
#fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
fStreamed = stream(fTest, cc, c)
print(fStreamed[1][0][0][1])


def collide(fArg, feqArg, psiArg, dtArg, tauArg):
    # fCollOut = np.zeros((len(fArg), len(fArg[0]), len(fArg[0][0]), len(fArg[0][0][0])), dtype=float)
    fCollOut = fArg - dtArg/tauArg * (fArg - feqArg) + (1.0 - dtArg / (2.0 * tauArg)) * dtArg * psiArg
    return fCollOut

# test collide
fTest = np.arange(m*n*o*27)
fTestEq = np.arange(m*n*o*27)
psiTest = np.arange(m*n*o*27)
dtTest = 0.1
tauTest = 0.55

fCollTest = collide(fTest, fTestEq, psiTest, dtTest, tauTest)
print(fCollTest.shape)


def sourceTermPsi(SArg, ccArg, wArg, csArg):
    psiOut = np.zeros((len(SArg), len(SArg[0]), len(SArg[0][0]), 27), dtype=float)
    for i in range(0, len(psiOut)):  # f0
        for j in range(0, len(psiOut[0])):
            for k in range(0, len(psiOut[0][0])):
                for l in range(0, len(psiOut[0][0][0])):
                    psiOut[i][j][k][l] = wArg[l] * 1.0 / (csArg ** 2) * np.tensordot(ccArg[l], SArg[i][j][k], axes=1)
    return psiOut


# test SourceTermPsi
STest = np.array( [ [[ [0.0, 1.0, 0.0], [2.0, 0.0, 0.0] , [0.0, 0.0, 3.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]]
                  ], dtype=float )

sourceTermPsiTest = sourceTermPsi(STest, cc, w, cs)
print(sourceTermPsiTest.shape)


def intitialize(rho0Arg, csArg, ccArg, wArg, mArg, nArg, oArg):
    P0 = np.zeros((mArg, nArg, oArg, 3, 3), dtype=float)
    j0 = np.zeros((mArg, nArg, oArg, 3), dtype=float)
    fOut = equilibriumDistribution(rho0Arg, j0, P0, ccArg, wArg, csArg)
    return [fOut, j0]


def computeRho(fArg):
    return zerothMoment(fArg)


def computeJ(fArg,SArg,ccArg, dtArg):
    jOut = firstMoment(fArg, ccArg) + dtArg/2.0 * SArg
    return jOut

# test computeJ
fTest = np.array([[[[0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12,12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20, 21.21, 22.22, 23.23, 24.24, 25.25, 26.26 ]]]], dtype = float)
STest = np.array( [ [[ [0.0, 1.0, 0.0], [2.0, 0.0, 0.0] , [0.0, 0.0, 3.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]], [[ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]], [ [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] , [0.0, 0.0, 0.0]]]
                  ], dtype=float )
jTest = computeJ(fTest, STest, cc, dtTest)
print(jTest.shape)


def computeP(fArg, ccArg):
    return secondMoment(fArg, ccArg)


def calculateMoments(fArg, SArg, ccArg, dtArg):
    return [computeRho(fArg), computeJ(fArg, SArg, ccArg, dtArg), computeP(fArg, ccArg)]


def computeU(uOldArg, rhoArg, jArg, dtArg):
    uNew = np.zeros((len(uOldArg), len(uOldArg[0]), len(uOldArg[0][0]), 3), dtype=float)
    for i in range(0, len(uOldArg)):  # f0
        for j in range(0, len(uOldArg[0])):
            for k in range(0, len(uOldArg[0][0])):
                uNew[i][j][k] = uOldArg[i][j][k] + jArg[i][j][k] / rhoArg[i][j][k] * dtArg
    return uNew


#test computeU
uOldTest = np.zeros((m, n, o, 3), dtype=float)
rhoTest = np.zeros((m, n, o), dtype=float)
rhoTest.fill(1.0)
jTest = np.zeros((m, n, o, 3), dtype=float)
jTest.fill(1.0)
dtTest = 0.1
uNewTest = computeU(uOldTest, rhoTest, jTest, dtTest)
print(uNewTest.shape)


def computeDivergenceU(uArg, dxArg):
    divUOut = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=float)
    uX = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=float)
    uY = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=float)
    uZ = np.zeros((len(uArg), len(uArg[0]), len(uArg[0][0])), dtype=float)


    for i in range(0, len(uArg)):
        for j in range(0, len(uArg[0])):
            for k in range(0, len(uArg[0][0])):
                uX[i][j][k] = uArg[i][j][k][0]
                uY[i][j][k] = uArg[i][j][k][1]
                uZ[i][j][k] = uArg[i][j][k][2]

    gradientUx = np.gradient(uX, dxArg, dxArg, dxArg)
    gradientUy = np.gradient(uY, dxArg, dxArg, dxArg)
    gradientUz = np.gradient(uZ, dxArg, dxArg, dxArg)

    for i in range(0, len(uArg)):
        for j in range(0, len(uArg[0])):
            for k in range(0, len(uArg[0][0])):
                divUOut[i][j][k] = gradientUx[0][i][j][k] + gradientUy[1][i][j][k] + gradientUz[2][i][j][k]

    return divUOut


# test computeDivergence
uTest = np.zeros((m, n, o, 3), dtype=float)
print(uTest.shape)
divUTest = computeDivergenceU(uTest, 0.1)
print(divUTest.shape)


def selectAtCoordinate(fArg, coordinateArg='x', coordinateValueArg=0):
    if coordinateArg == 'x':
        outF = fArg[coordinateValueArg]
    elif coordinateArg == 'y':
        outF = fArg[:, coordinateValueArg, :]
    elif coordinateArg == 'z':
        outF = fArg[:, :, coordinateValueArg]
    return outF

# test selectAtCoordinate
fTest = np.zeros((m, n, o, 27), dtype = float)
print(fTest.shape)
fSelectedTest = selectAtCoordinate(fTest, coordinateArg='z', coordinateValueArg=o-1)
print(fSelectedTest.shape)
fSelectedTest[0, 1, 0] = 989.0

# test select corner x=0, y=o, z=o

fXYZeq0 = selectAtCoordinate(selectAtCoordinate(selectAtCoordinate(fTest, coordinateArg='x', coordinateValueArg=0)))
print(fXYZeq0.shape)
fXYZeq0[5] = 5



def selectInterpolationNeighbors(arrayArg,  ccArg, cArg, coordinateArg='x', coordinateValueArg=0): # just normal to plane defined by coordinateArg = const
    arrayAtCoordinate = selectAtCoordinate(arrayArg, coordinateArg, coordinateValueArg)
    outNeighbors = np.zeros((len(arrayAtCoordinate), len(arrayAtCoordinate[0]), 27, 3), dtype=int) # saves i,j,k indices in arrayArg
    outNeighbors.fill(-1.0)
    for i in range(0, len(arrayAtCoordinate)):
        for j in range (0, len(arrayAtCoordinate[0])):
            for l in range(0, 27):
                cL = 1.0 / cArg * ccArg[l]
                if coordinateArg == 'x':
                    ii = coordinateValueArg
                    jj = i
                    kk = j
                elif coordinateArg == 'y':
                    ii = i
                    jj = coordinateValueArg
                    kk = j
                elif coordinateArg == 'z':
                    ii = i
                    jj = j
                    kk = coordinateValueArg

                potentialNeigborCoordinates = [int(ii + cL[0]), int(jj + cL[1]), int(kk + cL[2])]
                if checkIfIndicesInArrayBounds(potentialNeigborCoordinates[0], potentialNeigborCoordinates[1], potentialNeigborCoordinates[2], arrayArg):
                #if potentialNeigborCoordinates[0] < len(arrayArg) and potentialNeigborCoordinates[1] < len(arrayArg[0]) and potentialNeigborCoordinates[2] < len(arrayArg[0][0]):
                    outNeighbors[i,j,l] = np.array([potentialNeigborCoordinates[0], potentialNeigborCoordinates[1], potentialNeigborCoordinates[2]])
    return copy.deepcopy(outNeighbors)

# test select interpolation neighbors
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.4 , 15.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
rhoNeighborsTest = selectInterpolationNeighbors(rhoTest, cc, c, coordinateArg='z', coordinateValueArg=2)
print(rhoNeighborsTest.shape)


def extrapolateScalarToBd(rhoArg, ccArg, cArg, coordinateArg='x', coordinateValueArg=0):
    def interpolationNeighborExists(indicesArg):
        return indicesArg[0] != -1 and indicesArg[1] != -1 and indicesArg[2] != -1

    rhoBoundary = copy.deepcopy(selectAtCoordinate(rhoArg, coordinateArg, coordinateValueArg))
    IndicesInterpolationNeigbors = selectInterpolationNeighbors(rhoArg, ccArg, cArg, coordinateArg, coordinateValueArg)
    rhoBoundaryOut = np.zeros((len(rhoBoundary), len(rhoBoundary[0]), 27), dtype=float)
    for i in range(0, len(rhoBoundaryOut)):
        for j in range(0, len(rhoBoundaryOut[0])):
            for l in range(0, len(rhoBoundaryOut[0][0])):
                if(interpolationNeighborExists(IndicesInterpolationNeigbors[i,j,l])):
                    rhoBoundaryOut[i,j,l] = 0.5*(3.0 * rhoBoundary[i,j] - rhoArg[IndicesInterpolationNeigbors[i,j,l,0], IndicesInterpolationNeigbors[i,j,l,1], IndicesInterpolationNeigbors[i,j,l,2]])
    return rhoBoundaryOut

# test computeRhoBd
rhoTest = np.array( [ [[ 2.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 1.0], [ 1.0, 1.4 , 1.0]], [[ 1.0, 1.0 , 1.0], [ 1.0, 1.0 , 15.0], [ 1.0, 1.0 , 1.0]]
                  ], dtype=float )
rhoBdTest = extrapolateScalarToBd(rhoTest,cc, c, 'z', 2)
print(rhoBdTest)

# extrapolates second order tensor to boundary
def extrapolateTensorToBd(sigmaArg, ccArg, cArg, coordinateArg='x', coordinateValueArg=0):
    def interpolationNeighborExists(indicesArg):
        return indicesArg[0] != -1 and indicesArg[1] != -1 and indicesArg[2] != -1

    sigmaBoundary = copy.deepcopy(selectAtCoordinate(sigmaArg, coordinateArg, coordinateValueArg))
    IndicesInterpolationNeigbors = selectInterpolationNeighbors(sigmaArg, ccArg, cArg, coordinateArg, coordinateValueArg)
    sigmaBdOut = np.zeros((len(sigmaBoundary), len(sigmaBoundary[0]), 27, 3, 3), dtype=float)
    for i in range(0, len(sigmaBdOut)):
        for j in range(0, len(sigmaBdOut[0])):
            for l in range(0, len(sigmaBdOut[0][0])):
                if (interpolationNeighborExists(IndicesInterpolationNeigbors[i, j, l])):
                    sigmaBdOut[i, j, l] = 0.5 * (3.0 * sigmaBoundary[i, j] - sigmaArg[IndicesInterpolationNeigbors[i, j, l, 0], IndicesInterpolationNeigbors[i, j, l, 1],IndicesInterpolationNeigbors[i, j, l, 2]])
    return sigmaBdOut




# def computePBd(PArg, ccArg, cArg, coordinateArg='x', coordinateValueArg=0):
#     def interpolationNeighborExists(indicesArg):
#         return indicesArg[0] != -1 and indicesArg[1] != -1 and indicesArg[2] != -1
#     PBoundary = copy.deepcopy(selectAtCoordinate(PArg, coordinateArg, coordinateValueArg))
#     IndicesInterpolationNeigbors = selectInterpolationNeighbors(PArg, ccArg, cArg, coordinateArg, coordinateValueArg)
#     PBoundaryOut = np.zeros((len(PBoundary), len(PBoundary[0]), 27, 3, 3), dtype=float)
#     for i in range(0, len(PBoundaryOut)):
#         for j in range(0, len(PBoundaryOut[0])):
#             for l in range(0, len(PBoundaryOut[0][0])):
#                 if(interpolationNeighborExists(IndicesInterpolationNeigbors[i,j,l])):
#                     PBoundaryOut[i,j,l] = 0.5*(3.0 * PBoundary[i,j] - PArg[IndicesInterpolationNeigbors[i,j,l,0], IndicesInterpolationNeigbors[i,j,l,1], IndicesInterpolationNeigbors[i,j,l,2]])
#     return PBoundaryOut

# test computePBd
PTest = np.arange(3*3*3*3*3, dtype=float).reshape((3, 3, 3, 3, 3))
PBdTest = extrapolateTensorToBd(PTest, cc, c, coordinateArg='z', coordinateValueArg=2)
print(PBdTest.shape)


def getOppositeLatticeDirection(latticeDirection=0): # fits for Krueger convention D2Q27
    if (latticeDirection == 0):
        return 0
    elif (latticeDirection % 2 == 0):
        return (latticeDirection - 1)
    elif not (latticeDirection % 2 == 0):
        return (latticeDirection + 1)

# test getOppositeLatticeDirection
print(getOppositeLatticeDirection(3))


def getMissingDistributionFunctionIndices(fArg, coordinateArg='x', coordinateValueArg=0):
    if coordinateArg == 'x' and coordinateValueArg == 0:
        outIndices = [1, 7, 9, 13, 15, 19, 21, 23, 26]
    elif coordinateArg == 'x' and coordinateValueArg == len(fArg)-1:
        outIndices = [2, 8, 10, 14, 16, 20, 22, 24, 25]
    elif coordinateArg == 'y' and coordinateValueArg == 0:
        outIndices = [3, 7, 11, 14, 17, 19, 21, 24, 25]
    elif coordinateArg == 'y' and coordinateValueArg == len(fArg[0])-1:
        outIndices = [4, 8, 12, 13, 18, 20, 22, 23, 26]
    elif coordinateArg == 'z' and coordinateValueArg == 0:
        outIndices = [5, 9, 11, 16, 18, 19, 22, 23, 25]
    elif coordinateArg == 'z' and coordinateValueArg == len(fArg[0][0]) - 1:
        outIndices = [6, 10, 12, 15, 17, 20, 21, 24, 26]
    return outIndices



def applyDirichletBoundaryConditions(fArg, fCollArg, rhoArg, csArg, ccArg, cArg, wArg, uBdArg, coordinateArg='x', coordinateValueArg=0):  # TODO not defined how edges/corners are handled
    rhoBd = extrapolateScalarToBd(rhoArg, ccArg, cArg, coordinateArg, coordinateValueArg) # needs to be computed for lattice link
    jBd = np.zeros((len(rhoBd), len(rhoBd[0]), 27, 3), dtype=float)
    for i in range(0, len(jBd)):
        for j in range(0, len(jBd[0])):
            for l in range(0, 27):
                jBd[i,j,l] = uBdArg / rhoBd[i,j,l]

    #jBd = uBdArg / rhoBd # interpolated at all lattice link directions
    fCollRelevant = selectAtCoordinate(fCollArg, coordinateArg, coordinateValueArg)
    fRelevant = copy.deepcopy(selectAtCoordinate(fArg, coordinateArg, coordinateValueArg))
    indicesMissing = getMissingDistributionFunctionIndices(fArg, coordinateArg, coordinateValueArg)
    for i in range(0, len(fRelevant)):
        for j in range(0, len(fRelevant[0])):
            for l in indicesMissing:
                oL = getOppositeLatticeDirection(l)
                test = jBd[i,j,l]
                test2 = np.tensordot(ccArg[oL], jBd[i,j,l],axes=1)
                #test2 = ccArg[oL]
                fRelevant[i,j,l] = fCollRelevant[i,j,oL] - 2.0 / csArg ** 2 * wArg[oL] * np.tensordot(ccArg[oL], jBd[i,j,l],axes=1)   # l goes into domain from boundary -> is the correct position to interpolate jBd from, oL streams across boundary
    return fRelevant


# test dirichlet BC
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(np.nan)
fCollTest = np.arange(m*n*o*27, dtype = float).reshape((m,n,o,27))
rhoTest = np.zeros((m, n, o), dtype = float)
rhoTest.fill(1.0)
uBdTest = np.array([1,0,0])
fTestApplied = applyDirichletBoundaryConditions(fTest, fCollTest, rhoTest, cs, cc, c, w, uBdTest)
print(fTestApplied.shape)


def computeSigma(PArg, divUArg, laArg, mueArg):
    sigmaOut = np.zeros(PArg.shape, dtype=float) # TODO how is sigma computed?
    for i in range(0, len(sigmaOut)):
        for j in range(0, len(sigmaOut[0])):
            for k in range(0, len(sigmaOut[0][0])):
                sigmaOut[i,j,k] = PArg[i, j, k] + (laArg-mueArg) * divUArg[i, j, k] * np.identity(3, dtype=float)
    return sigmaOut


# test computeSigma
PTest = np.arange(m*n*o*3*3, dtype=float).reshape((m, n, o, 3, 3))
divUTest = np.arange(m*n*o, dtype=float).reshape((m, n, o))
laTest = 1.5
mueTest = 1.0
sigmaTest = computeSigma(PTest, divUTest, laTest, mueTest)
print(sigmaTest.shape)


def computePBd(sigmaBC, fArg, ccArg, cArg, uArg, dxArg, laArg, mueArg, coordinateArg='x', coordinateValueArg=0):
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


    P = secondMoment(fArg, ccArg)
    divU = computeDivergenceU(uArg, dxArg)
    sigma = computeSigma(P, divU, laArg, mueArg)
    sigmaBd = extrapolateTensorToBd(sigma,ccArg,cArg,coordinateArg,coordinateValueArg)
    sigmaBd = applyTractionBoundaryConditions(sigmaBd, sigmaBC)
    divUBd = extrapolateScalarToBd(divU, ccArg, cArg, coordinateArg, coordinateValueArg)
    Pbd = np.zeros(sigmaBd.shape, dtype=float)
    for i in range(0, len(sigmaBd)):
        for j in range(0, len(sigmaBd[i])):
            for l in range(0, len(sigmaBd[i][j])):
                Pbd[i,j,l] = -sigmaBd[i,j,l] + (laArg - mueArg) * divUBd[i,j,l] * np.identity(3, dtype=float)
    return Pbd


# test computePBd
sigmaBCTest = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, np.nan], [1.0, np.nan, np.nan]], dtype=float)
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(0.5)
uTest = np.arange(m*n*o*3, dtype=float).reshape((m,n,o,3))
PBdTest = computePBd(sigmaBCTest,fTest,cc,c,uTest,0.1,1.5,1.0,'x',m-1)
print(PBdTest.shape)


def applyNeumannBoundaryConditions(fArg, fCollArg, uArg , rhoArg, csArg, ccArg, cArg, wArg, sigmaBdArg, dxArg, laArg, mueArg, coordinateArg='x', coordinateValueArg=0):
    rhoBd = extrapolateScalarToBd(rhoArg, ccArg, cArg, coordinateArg,
                                  coordinateValueArg)  # needs to be computed for lattice link


    # jBd = uBdArg / rhoBd # interpolated at all lattice link directions
    fCollRelevant = selectAtCoordinate(fCollArg, coordinateArg, coordinateValueArg)
    fRelevant = copy.deepcopy(selectAtCoordinate(fArg, coordinateArg, coordinateValueArg))
    indicesMissing = getMissingDistributionFunctionIndices(fArg, coordinateArg, coordinateValueArg)
    PBd = computePBd(sigmaBdArg,fArg,ccArg,cArg,uArg,dxArg,laArg,mueArg,coordinateArg,coordinateValueArg)
    for i in range(0, len(fRelevant)):
        for j in range(0, len(fRelevant[0])):
            for l in indicesMissing:
                oL = getOppositeLatticeDirection(l)

                tmp1 = PBd[i,j,l] - rhoBd[i,j,l] * csArg ** 2 * np.identity(3,dtype=float)
                tmp2 = np.outer(ccArg[oL], ccArg[oL].transpose()) - csArg ** 2 * np.identity(3, dtype=float)


                #tmp2 = np.tensordot((PArg[i][j][k] - rhoArg[i][j][k] * cs ** 2 * np.identity(3, dtype=float)),
                #                    (np.outer(ccArg[l], ccArg[l].transpose()) - cs ** 2 * np.identity(3, dtype=float)),
                #                    axes=2)
                #test = jBd[i, j, l]
                #test2 = np.tensordot(ccArg[oL], jBd[i, j, l], axes=1)
                # test2 = ccArg[oL]


                fRelevant[i,j,l] = - fCollRelevant[i, j, oL] + 2.0 * wArg[oL] * (rhoBd[i,j,l] + 1.0 / (2.0 * csArg ** 4) * (np.tensordot(tmp1,tmp2, axes=2)))


                #fRelevant[i, j, l] = fCollRelevant[i, j, oL] - 2.0 / csArg ** 2 * wArg[oL] * np.tensordot(ccArg[oL],
                #                                                                                          jBd[i, j, l],
                #                                                                                          axes=1)  # l goes into domain from boundary -> is the correct position to interpolate jBd from, oL streams across boundary
    return fRelevant


# def computePBd(PArg, ccArg, cArg, coordinateArg='x', coordinateValueArg=0):
#     def interpolationNeighborExists(indicesArg):
#         return indicesArg[0] != -1 and indicesArg[1] != -1 and indicesArg[2] != -1
#     PBoundary = copy.deepcopy(selectAtCoordinate(PArg, coordinateArg, coordinateValueArg))
#     IndicesInterpolationNeigbors = selectInterpolationNeighbors(PArg, ccArg, cArg, coordinateArg, coordinateValueArg)
#     PBoundaryOut = np.zeros((len(PBoundary), len(PBoundary[0]), 27, 3, 3), dtype=float)
#     for i in range(0, len(PBoundaryOut)):
#         for j in range(0, len(PBoundaryOut[0])):
#             for l in range(0, len(PBoundaryOut[0][0])):
#                 if(interpolationNeighborExists(IndicesInterpolationNeigbors[i,j,l])):
#                     PBoundaryOut[i,j,l] = 0.5*(3.0 * PBoundary[i,j] - PArg[IndicesInterpolationNeigbors[i,j,l,0], IndicesInterpolationNeigbors[i,j,l,1], IndicesInterpolationNeigbors[i,j,l,2]])
#     return PBoundaryOut

# test Neumann BC
fTest = np.zeros((m, n, o, 27), dtype = float)
fTest.fill(0.5)
fCollTest = np.arange(m*n*o*27, dtype = float).reshape((m,n,o,27))
rhoTest = np.zeros((m, n, o), dtype = float)
rhoTest.fill(1.0)
sigmaBCTest = np.array([[1.0, 1.0, 1.0], [1.0, np.nan, np.nan], [1.0, np.nan, np.nan]], dtype=float)
uTest = np.arange(m*n*o*3, dtype=float).reshape((m,n,o,3))
uBdTest = np.array([1,0,0])

fTestApplied = applyNeumannBoundaryConditions(fTest,fCollTest,uTest,rhoTest,cs,cc,c,w,sigmaBCTest,0.1,1.5,1.9,'x',m-1)
#fTestApplied = applyDirichletBoundaryConditions(fTest, fCollTest, rhoTest, cs, cc, c, w, uBdTest)
print(fTestApplied.shape)


def writeVTK(k,name,path, points, time):
    file_name = path + name + '.vtk.' + '{:06d}'.format(k)
    my_file = open(file_name, 'w')

    my_file.write('# vtk DataFile Version 2.0 \n')
    my_file.write(
        'generated by Lattice Boltzmann Method time = {0:16.8e}\n'.format(time))
    my_file.write('ASCII \n')
    my_file.write('DATASET UNSTRUCTURED_GRID\n')
    my_file.write('\n')

    tmp_point = points[0]
    fields_in_PPData_number = len(tmp_point.PPData.keys())
    point_number = len(points)

    my_file.write('POINTS {0:6d} FLOAT \n'.format(point_number))
    for point in points:
        positionString = '{0:16.8e} '.format(point.x) + '{0:16.8e} '.format(
            point.y) + '{0:16.8e} \n'.format(point.z)
        my_file.write(positionString)
    my_file.write('\n')  # optional?

    my_file.write('CELLS {0:6d} {1:6d}\n'.format(point_number, 2 * point_number))
    for i in range(0, len(points)):
        connectivityString = '1 {0:6d} \n'.format(i)
        my_file.write(connectivityString)

    my_file.write('CELL_TYPES {0:6d}\n'.format(point_number))
    for i in range(0, len(points)):
        cellTypeString = '1\n'
        my_file.write(cellTypeString)

    my_file.write('POINT_DATA {0:6d}\n'.format(point_number))
    my_file.write('FIELD solution {0:1d}\n'.format(fields_in_PPData_number))

    def write_PPData_2_vtk(point, key):
        if hasattr(point.PPData[key], "__len__"):  # if PPData[key] is a list
            tmp_string = ''
            for i in range(0, len(point.PPData[key])):
                tmp_string2 = '{0' + ':18.8e} '
                value = point.PPData[key][i]
                if (value is None):
                    value = -999
                tmp_string = tmp_string + tmp_string2.format(value)
        else:
            value = point.PPData[key]
            if (value is None):
                value = -999
            tmp_string = '{0:18.8e} '.format(value)
        return tmp_string + '\n'

    for key in tmp_point.PPData.keys():
        if hasattr(tmp_point.PPData[key], '__len__'):
            # tmp_len = str(fields_in_PPData_number)  # can deal with arrays
            tmp_len = str(int(len(tmp_point.PPData[key])))
        else:
            tmp_len = str(1)

        my_file.write(key + ' ' + tmp_len + ' {0:6d} DOUBLE\n'.format(point_number))
        for point in points:
            my_file.write(write_PPData_2_vtk(point,key))
        my_file.write('\n')

    my_file.close()


# test writeVTK
class Point(object):
    def __init__(self, x, y, z, PPData = dict()):
        self.x = x
        self.y = y
        self.z = z
        self.PPData = PPData


xTest = np.zeros((m, n, o, 3), dtype = float)
dx = 0.1
points = list()

for i in range(0, len(xTest)):
    for j in range(0,len(xTest[0])):
        for k in range(0, len(xTest[0][0])):
            xTest[i,j,k] = np.array([float(i) *dx, float(j) * dx, float(k) * dx], dtype=float)
            ppData = dict()
            ppData["u"] = [1.0,0.0,0.0]
            points.append(Point(float(i) *dx, float(j) * dx, float(k) * dx, ppData))

writeVTK(1,"test","/Users/alex/Desktop/",points,0.1)




print("End")