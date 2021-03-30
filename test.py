import numpy as np

class Test:
    def __init__(self):
        self.a = np.array([1,2])
        self.b = np.array([3,4,5])
        self.c = np.array([7, 8, 9, 10])

    def get_all_param(self):
        return self.a, self.b, self.c

if __name__=="__main__":
    # test = Test()
    # print(type(test.get_all_param()))
    a = np.array([1, 2, 3, 40, 5])
    b = np.array([10, 10, 10, 10, 10])
    c = 10
    d = np.tile(a, (5, 1))
    print(a < b)
    print(np.all(a<b))
    print(a < c)
    print(np.all(a<c))
    print(d < c)