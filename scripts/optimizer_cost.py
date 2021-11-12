import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares

def fun():

    v1 = lambda x: np.linalg.norm(x)
    return v1


def con1(args):
    t_min, t_max, J, W = args # J:6*7, W:6*1
    J = np.array(J)
    W = np.array(W)
    cons = [{'type': 'ineq', 'fun': lambda x: x[0] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[1] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[2] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[3] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[4] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[5] - t_min},
            {'type': 'ineq', 'fun': lambda x: x[6] - t_min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[1] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[2] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[3] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[4] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[5] + t_max},
            {'type': 'ineq', 'fun': lambda x: -x[6] + t_max},
            {'type': 'eq', 'fun': lambda x: J@x-W}
            ]
    # for i in range(6):
    #     j = J[i].reshape((1,7))
    #     con = {'type': 'eq', 'fun': lambda x, i=i: j[0]@x-W[i]}
    #     cons.append(con)

    return cons


def minimize_cost(args):
    cons = con1(args)
    x0 = np.asarray((2.5,1.5,1.5,1.5,1.5,1.5,1.5))
    res = minimize(fun(), x0, method='SLSQP', constraints=cons,options={'disp':True})
    v = res.x  # 返回最小化后的变量
    result = res.fun # 返回最小化后的结果
    print('*******************')
    print('cost_result',result)
    print(res.x)
    print()
    print()
    print(res.success)
    print()
    return v

if __name__ == "__main__":
    t_min, t_max = 0.1,5
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
    # print(np.shape(J[0]))
    # j = np.array(J[0]).reshape((1,7))
    # print(j)
    args = (t_min, t_max, J, W)
    v = minimize_cost(args)
    # print(J@v-W)

