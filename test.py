import numpy as np

class Test:
    def __init__(self):
        self.a = np.array([1,2])
        self.b = np.array([3,4,5])
        self.c = np.array([7, 8, 9, 10])

    def get_all_param(self):
        return [self.a, self.b, self.c]

if __name__=="__main__":
    test = Test()
    chi = np.array(test.get_all_param())