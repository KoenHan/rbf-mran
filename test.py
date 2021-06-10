import time
import numpy as np

a = np.array([i for i in range(25)]).reshape((5, 5))
print(a)
b = a.reshape((1, -1))
print(b)
c = b[0].tolist()
print(c)
# c = b.reshape((5, -1))
# print(c)
