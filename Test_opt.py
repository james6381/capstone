from cvxopt import *
from cvxopt.modeling import *
import numpy as np
import cvxpy

# Testing Lin Solver
# MIN x1 + 4x2 - 4x3 +5x4
# st -x1 + x2 <= 1
#     x2 + x3 = 2
#     x3 + x4 = 5
#     x3 >= 0
#     x4 >= 1

c = np.zeros((1,4))
A = np.zeros((3,4))
b = np.zeros((3,1))
Aeq = np.zeros((2,4))
beq = np.zeros((2,1))

# WILL NOT WORK WITH np.matrix
c = np.array([1.0, 4.0, -4.0, 5.0])
A = np.array([ [-1.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, -1.0] ])
b = np.array([1.0, 0.0, -1.0])
Aeq = np.array([ [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0] ])
beq = np.array([2.0, 5.0])


c1 = matrix(c)
A1 = matrix(A)
b1 = matrix(b)
Aeq1 = matrix(Aeq)
beq1 = matrix(beq)

sol = solvers.lp(c1, A1, b1, Aeq1, beq1)
print(sol['x'])