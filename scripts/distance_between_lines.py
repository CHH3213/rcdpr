import math
import numpy as np

'''
求空间中两条线段的距离
@chh
'''

'''
法一,clampAll=False：垂直的最段距离； 
    clampAll=True：变成点到线的最短距离，与法2一样
'''

def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False, clampB1=False):
    """
    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance
    :param a0: line1 one side
    :param a1: line1 other side
    :param b0: line2 one side
    :param b1: line2 other side
    :param clampAll:
    :param clampA0:
    :param clampA1:
    :param clampB0:
    :param clampB1:
    :return: point1 in line1, point2 in line2, shortest distance
    """
    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denominator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2
    # print(denom)
    '''
    # If lines are parallel (denom=0), then test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    '''
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)

            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)
        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)
    return pA, pB, np.linalg.norm(pA - pB)


'''
法二
'''
def dot(c1, c2):
    return c1[0] * c2[0] + c1[1] * c2[1] + c1[2] * c2[2]


def getShortestDistance(a0, a1, b0, b1):
    x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4 = a0[0], a1[0], b0[0], b1[0], a0[1], a1[1], b0[1], b1[1], a0[2], a1[
        2], b0[2], b1[2]
    # print(x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4)
    EPS = 0.00000001
    delta21 = [1, 2, 3]
    delta21[0] = x2 - x1
    delta21[1] = y2 - y1
    delta21[2] = z2 - z1

    delta41 = [1, 2, 3]
    delta41[0] = x4 - x3
    delta41[1] = y4 - y3
    delta41[2] = z4 - z3

    delta13 = [1, 2, 3]
    delta13[0] = x1 - x3
    delta13[1] = y1 - y3
    delta13[2] = z1 - z3

    a = dot(delta21, delta21)
    b = dot(delta21, delta41)
    c = dot(delta41, delta41)
    d = dot(delta21, delta13)
    e = dot(delta41, delta13)
    D = a * c - b * b

    sc = D
    sN = D
    sD = D
    tc = D
    tN = D
    tD = D

    if D < EPS:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a

    elif tN > tD:
        tN = tD
        if ((-d + b) < 0.0):
            sN = 0
        elif ((-d + b) > a):
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    if (abs(sN) < EPS):
        sc = 0.0
    else:
        sc = sN / sD
    if (abs(tN) < EPS):
        tc = 0.0
    else:
        tc = tN / tD

    dP = [1, 2, 3]
    dP[0] = delta13[0] + (sc * delta21[0]) - (tc * delta41[0])
    dP[1] = delta13[1] + (sc * delta21[1]) - (tc * delta41[1])
    dP[2] = delta13[2] + (sc * delta21[2]) - (tc * delta41[2])

    return math.sqrt(dot(dP, dP))


def is_parallel(vec1, vec2):
    """ 判断二个三维向量是否平行 """
    assert isinstance(vec1, np.ndarray), r'输入的 vec1 必须为 ndarray 类型'
    assert isinstance(vec2, np.ndarray), r'输入的 vec2 必须为 ndarray 类型'
    assert vec1.shape == vec2.shape, r'输入的参数 shape 必须相同'

    vec1_normalized = vec1 / np.linalg.norm(vec1)
    vec2_normalized = vec2 / np.linalg.norm(vec2)

    if 1.0 - abs(np.dot(vec1_normalized, vec2_normalized)) < 1e-6:
        return True
    else:
        return False


if __name__ == '__main__':
    # a1=np.array([13.43, 21.77, 46.81])
    # a0=np.array([27.83, 31.74, -26.60])
    # b0=np.array([77.54, 7.53, 6.22])
    # b1=np.array([26.99, 12.39, 11.18])

    # 测试在平行的情况：
    a0 = np.array([1, 0, 1])
    a1 = np.array([0, 0, 0])
    b0 = np.array([1, 1, 0])
    b1 = np.array([0, 1, 0])

    print(getShortestDistance(a0, a1, b0, b1))
    print(closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False))

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([-1.0, 0.0, 0.0])
    print(is_parallel(vec1, vec2))

    vec3 = np.array([0.5, 0.0, 0.0])
    print(is_parallel(vec1, vec3))

    vec4 = np.array([0.0, 1.0, 0.0])
    print(is_parallel(vec1, vec4))
