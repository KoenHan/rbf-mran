import numpy as np

class Test:
    def __init__(self):
        self.a = np.array([1,2])
        self.b = np.array([3,4,5])
        self.c = np.array([7, 8, 9, 10])

    def get_all_param(self):
        return self.a, self.b, self.c

if __name__=="__main__":
    # a = np.array([i for i in range(30)]).reshape(3, 10)
    # print(a)
    # a = np.delete(a, slice(3,5), 1)
    # print(a)

    # with open('./data/siso/train.txt', mode='r') as f:
    #     l = f.readlines()
    # l = [list(map(float, s.strip().split())) for s in l]
    # # l = list(map(float, l))
    # # print(l)
    # print(type(l[0][0]))
    # print(l[0].strip().split())
    # print(l[3].strip().split())
    l = [[0, 1], [2, 3], [4, 5]]
    print(l[:][0])