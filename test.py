import numpy as np
import cupy as cp
import time

A_cpu = np.random.rand(2000, 2000)
B_cpu = np.random.rand(2000, 2000)
start = time.time()
AB_cpu = np.dot(A_cpu, B_cpu)
duration = time.time() - start
print(duration)

A_gpu = cp.random.rand(2000, 2000)
B_gpu = cp.random.rand(2000, 2000)

start = time.time()
AB_gpu = cp.dot(A_gpu, B_gpu)
duration = time.time() - start
AB_cpu2 = AB_gpu.get()  # AB_cpu2 は np.ndarray 型
print(duration)

print(cp.cuda.runtime.runtimeGetVersion())
from cupy.cuda import cudnn
print(cudnn.getVersion())