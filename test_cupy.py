# https://tutorials.chainer.org/ja/10_Introduction_to_CuPy.html
# から抜粋した比較

import time
import cupy as cp
import numpy as np

def get_w_np(x, t):
    xx = np.dot(x.T, x)
    xx_inv = np.linalg.inv(xx)
    xt = np.dot(x.T, t)
    w = np.dot(xx_inv, xt)
    return w

def get_w_cp(x, t):
    xx = cp.dot(x.T, x)
    xx_inv = cp.linalg.inv(xx)
    xt = cp.dot(x.T, t)
    w = cp.dot(xx_inv, xt)
    return w


# for N in [10, 100, 1000, 10000]: # これでやるとN=100の時だけ異常に速くなる
N = 100
x = np.random.rand(N, N)
t = np.random.rand(N, 1)

s1 = time.time()
w = get_w_np(x, t)
e1 = time.time()
print('N = ', str(N))
print(e1 - s1)

# x_cp = cp.asarray(x)
# t_cp = cp.asarray(t)
cp.random.seed(0)
x_cp = cp.random.rand(N, N)
t_cp = cp.random.rand(N, 1)

cp.cuda.Stream.null.synchronize()
s2 = time.time()

w = get_w_cp(x_cp, t_cp)

cp.cuda.Stream.null.synchronize()
e2 = time.time()
print(e2 - s2)

'''
N = 10
0.00024247169494628906
0.2500169277191162
N = 100
0.0005815029144287109
0.24589943885803223
N = 1000
0.04524564743041992
0.3386847972869873
N = 10000
25.85020422935486
34.97923493385315
'''
