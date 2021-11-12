'''
使用scipy方式优化 4uav,3ugv 力
'''
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import least_squares


def fun(args):
    target_force, force_dir0, force_dir1, force_dir2, force_dir3, force_dir4, force_dir5, force_dir6 = args
    target_force = np.array(target_force)
    force_dir0 = np.array(force_dir0)
    force_dir1 = np.array(force_dir1)
    force_dir2 = np.array(force_dir2)
    force_dir3 = np.array(force_dir3)
    force_dir4 = np.array(force_dir4)
    force_dir5 = np.array(force_dir5)
    force_dir6 = np.array(force_dir6)
    # 最小化模
    v1 = lambda x: np.linalg.norm(
        x[0] * force_dir0 + x[1] * force_dir1 + x[2] * force_dir2 + x[3] * force_dir3 + x[4] * force_dir4 + x[
            5] * force_dir5 + x[6] * force_dir6 - target_force)
    # 最小化模与最大化余弦距离
    # v1 = lambda x: np.linalg.norm(x[0]*force_dir0 + x[1]*force_dir1 + x[2]*force_dir2 + x[3]*force_dir3 + x[4]*force_dir4 + x[5]*force_dir5 - target_force)-np.dot(target_force,x[0]*force_dir0 + x[1]*force_dir1 + x[2]*force_dir2 + x[3]*force_dir3 + x[4]*force_dir4 + x[5]*force_dir5)/(np.linalg.norm((x[0]*force_dir0 + x[1]*force_dir1 + x[2]*force_dir2 + x[3]*force_dir3+ x[4]*force_dir4 + x[5]*force_dir5))*np.linalg.norm(target_force))
    return v1


def con1(args):
    vmin, vmax = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[1] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[2] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[3] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[4] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[5] - vmin},
            {'type': 'ineq', 'fun': lambda x: x[6] - vmin},
            {'type': 'ineq', 'fun': lambda x: -x[0] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[1] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[2] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[3] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[4] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[5] + vmax},
            {'type': 'ineq', 'fun': lambda x: -x[6] + vmax}
            )
    return cons


def minimizeForce(allArgs):
    # target_force, force_dir0, force_dir1,force_dir2,force_dir3,force_dir4,force_dir5 = allArgs
    # 力的下限是大于0,因为是小车的拉力，方向指向小车
    args = [0, 5000] 
    cons = con1(args)

    x0 = np.asarray((1, 1, 1, 1, 1, 1, 1))
    res = minimize(fun(allArgs), x0, method='SLSQP', constraints=cons)
    v = res.x  # 返回最小化后的变量
    result = res.fun  # 返回最小化后的结果
    print('result', result)
    # print(res.x)
    print(res.success)
    return v, result


if __name__ == "__main__":
    import time

    s = time.time()
    args = ([0, 0, 2.5], [0, 4, 4.5], [3.464, -2, 4.5],
            [-3.464, -2, 4.5], [0, 0, 4.5], [0, 2, 0], [1.732, -1, 0], [-1.732, -1, 0])
    v, result = minimizeForce(args)
    print(v)
    print(time.time() - s)
