"""
Description: FEM implementation of block under tension example, model adapted
from https://fenics-solid-tutorial.readthedocs.io/en/latest/Elastodynamics/Elastodynamics.html
"""

import sys
import dolfin as dlf
import numpy as np
from time import time

################################################################################
    # Parameters
################################################################################
#number of nodes in mesh
n = 10

t0 = 0.01 #max traction value for NeumannBC

#write file and put out y displacement from reference points
output = True

# Time-stepping
t_start = 0.0
t_end = 4.0
dt = 0.02887

#Material Parameters / lame constants
lmbda = 1.0
mu = 1.0
rho = 1.0

################################################################################
    # Model Building
################################################################################

# bot nodes
def bot(x, on_boundary):
    return (on_boundary and dlf.near(x[1], -1.0))

# top nodes
def top(x, on_boundary):
    return (on_boundary and dlf.near(x[1], 1.0))

# Strain function
def eps(u):
    return 0.5*(dlf.nabla_grad(u) + dlf.nabla_grad(u).T)

# Stress function
def sig(u):
    return lmbda*dlf.div(u)*dlf.Identity(3) + 2*mu*eps(u)


# --------------------
# Geometry
# --------------------
mesh = dlf.BoxMesh(dlf.Point(-1.0, -1.0, -1.0), dlf.Point(1.0, 1.0, 1.0), n, n, n)
print("Number of nodes: {}".format(len(mesh.coordinates())))

# --------------------
# Function spaces
# --------------------
V = dlf.VectorFunctionSpace(mesh, "CG", 1)
u_tr = dlf.TrialFunction(V)
u_test = dlf.TestFunction(V)

#NeumannBC at top and bot
traction_markers = dlf.MeshFunction('size_t', mesh, mesh.topology().dim()-1)  # on edges in 2D
traction_markers.set_all(0)
dlf.AutoSubDomain(top).mark(traction_markers, 1) # 1 for top edge with traction
dlf.AutoSubDomain(bot).mark(traction_markers, 2) # 2 for top edge with traction

dst = dlf.Measure('ds', domain=mesh, subdomain_data=traction_markers)(1)
dsb = dlf.Measure('ds', domain=mesh, subdomain_data=traction_markers)(2)

#expression defines traction on boundaries -> c++ syntax as a string
tDt = dlf.Expression(("0"," t < 2.0 ? (t/2.0)*t0 : t0","0"), t0 = t0, t=0.0, degree=2)
tDb = dlf.Expression(("0"," t < 2.0 ? -(t/2.0)*t0 : -t0","0"), t0 = t0, t=0.0, degree=2)

#DirichletBC -> not necessary for this example
bc = []

u = dlf.Function(V)

uo = dlf.Function(V)
uoo = dlf.Function(V)

#writing output in xdmf
file = dlf.XDMFFile("block_3D_out.xdmf")  # XDMF file

#weak form
A_form = dlf.inner(sig(u_tr), eps(u_test))*dlf.dx + rho/(dt*dt)*dlf.dot(u_tr - 2*uo + uoo, u_test)*dlf.dx + dlf.dot(tDt, u_test)*dst + dlf.dot(tDb, u_test)*dsb

t, i = 0,0
t_s = time()
#while t < t_end:
for i in range(150):
    tDt.t = t
    tDb.t = t

    dlf.solve(dlf.lhs(A_form) == dlf.rhs(A_form), u, bc, solver_parameters={"linear_solver": "superlu"})

    #Euler forward (explicit time integration)
    uoo.assign(uo)
    uo.assign(u)

    #optional output
    if output:
        file.write(u, t)
        print("p0", t, u(0.025,0.975,0.025)[1],  u(0.475,0.975,0.025)[1],  u(0.975,0.975,0.025)[1])

    #increae total time
    t +=dt
    i += 1

    print("Iteration: {}, time: {}".format(i,t))
dlf.list_timings(dlf.cpp.common.TimingClear.clear,[dlf.TimingType.wall])
t_e = time()
print("Total time: {}".format(t_e-t_s))
file.close()
dlf.list_linear_solver_methods()
