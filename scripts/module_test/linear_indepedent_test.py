# -*-coding:utf-8-*-
import sympy
import numpy as np
'''
如何从矩阵中查找线性独立的行
'''
mat = np.array([[0,1,0,0,1],
                [0,0,1,0,2],
                [0,1,1,0,3],
                [1,0,0,1,4]])
_, inds = sympy.Matrix(mat).rref()
print(np.shape(mat))
print(inds)

# print(mat.T)