import numpy as np
import cupy as cp
import time

n = 10000
A_cpu = np.random.rand(n, n)
B_cpu = np.random.rand(n, n)

start = time.time()
AB_cpu = np.dot(A_cpu, B_cpu)
duration = time.time() - start
print(duration)


cp.cuda.Stream.null.synchronize()
start = time.time()

# pattern A
A_gpu = cp.array(A_cpu)
B_gpu = cp.array(B_cpu)
# pattern B
# A_gpu = cp.random.rand(n, n)
# B_gpu = cp.random.rand(n, n)
AB_gpu = cp.dot(A_gpu, B_gpu)

# cp.cuda.Stream.null.synchronize()
finish = time.time()
print(finish - start)

AB_cpu2 = AB_gpu.get()  # AB_cpu2 は np.ndarray 型
