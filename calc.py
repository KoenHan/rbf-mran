import numpy as np

if __name__=="__main__" :
    a = np.loadtxt('ros_test_after_test.txt')
    # print(a)
    sum = [0., 0., 0.]
    for i in range(3):
        for item in a[i]:
            sum[i] += item
    print(sum)