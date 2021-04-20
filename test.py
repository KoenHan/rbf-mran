import numpy as np
import time

N = 2
x = -np.random.rand(N, N)

start = time.time()
a = abs(x)
end = time.time()
print(end - start)
print(x)
print(a)


x = -np.random.rand(N, N)
start = time.time()
b = np.abs(x)
end = time.time()
print(end - start)
print(x)
print(b)
