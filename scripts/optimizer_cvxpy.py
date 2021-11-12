'''
使用cvxpy库进行Jt=W的求解'''
import cvxpy as cp
from cvxpy.reductions import solvers
import numpy as np
import time


def minimizeForce(args):
    J, W = args  # J:6*7, W:6*1
    J = np.array(J)
    W = np.array(W)
    x = cp.Variable(7)
    obj = cp.Minimize(cp.norm(x))
    # t_min = np.array([0.1 for _ in range(7)])
    # t_max = np.array([5 for _ in range(7)])
    t_min = 0.1
    t_max = 5
    # con = [x[0] - t_min >= 0, x[1] - t_min >= 0, x[2] - t_min >= 0,
    #        x[3] - t_min >= 0, x[4] - t_min >= 0, x[5] - t_min >= 0,
    #        x[0] - t_max <= 0, x[1] - t_max <= 0, x[2] - t_max <= 0,
    #        x[3] - t_max <= 0, x[4] - t_max <= 0, x[5] - t_max <= 0 ]
    con = [x - t_min >= 0, x - t_max <= 0, J @ x == W]
    proble = cp.Problem(obj, con)
    proble.solve(solver=cp.SCS)
    print(x.value)
    print('status', proble.status)
    print('optimize_result', proble.value)

    return x.value, proble.value


if __name__ == '__main__':
    print(cp.installed_solvers())  # 已安装优化器
    J = [[0.00000000e+00, 3.86145729e-01, - 3.86145729e-01, 0.00000000e+00,
          0.00000000e+00, 8.65563252e-01, - 8.65563252e-01],
         [4.45867437e-01, - 2.22988946e-01, - 2.22988946e-01, 0.00000000e+00,
          9.99573291e-01, - 4.99947465e-01, - 4.99947465e-01],
         [8.95099005e-01,
          8.95079553e-01,
          8.95079553e-01,
          1.00000000e+00,
          2.92102072e-02,
          2.92025389e-02,
          2.92025389e-02],
         [2.36390241e-01, - 1.17742008e-01, - 1.17742008e-01, 0.00000000e+00,
          5.84204144e-02, - 2.92025389e-02, - 2.92025389e-02],
         [0.00000000e+00, - 2.04462602e-01,
          2.04462602e-01,
          0.00000000e+00,
          - 0.00000000e+00, - 5.05787973e-02,
          5.05787973e-02],
         [0.00000000e+00, - 1.42251569e-04, 1.42251569e-04, 0.00000000e+00,
          0.00000000e+00, - 3.45758060e-04, 3.45758060e-04]]
    W = [-4.44089210e-16, 0.00000000e+00, -1.00000000e+00, -1.24900090e-16,
         -3.05311332e-16, 0.00000000e+00]

    args = (J,W)
    s = time.time()
    v, result = minimizeForce(args)
    print(v)
    print(time.time() - s)
