import numpy as np
import random

def siso(pre_x, pre_u):
    y = 29/40*np.sin((16*pre_u + 8*pre_x)/(3 + 4*pre_u**2 + 4*pre_x**2))
    y += 0.2*(pre_u + pre_x)
    return y

if __name__ == "__main__":
    pre_x = 0
    pre_u = -1
    data_len = 5000
    x = 0
    file_name = ['./data/train.txt', './data/val.txt']
    for fn in file_name :
        with open(fn, mode='w') as f:
            for n in range(data_len):
                s = []
                # for _ in range(3):
                #     s.append(str(random.randint(0, 10)/10))
                if n :
                    x = siso(pre_x, pre_u)
                u = random.uniform(-1, 1)
                s.append(str(x))
                s.append(str(u))
                pre_u = u
                pre_x = x

                f.write('\t'.join(s)+'\n')