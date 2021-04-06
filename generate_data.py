import numpy as np
import random

class SISO:
    """
    p.54 BM-1
    """
    def __init__(self):
        self._pre_x = 0
        self._pre_u = -1
    
    def get_nx_and_nu(self):
        return {'nx':1, 'nu':1}

    def get_x(self):
        x = 29/40*np.sin((16*self._pre_u + 8*self._pre_x)/(3 + 4*self._pre_u**2 + 4*self._pre_x**2))
        x += 0.2*(self._pre_u + self._pre_x)
        return x

    def get_u(self):
        return random.uniform(-1, 1)

    def set_pre_x_and_u(self, x, u):
        self._pre_x = x
        self._pre_u = u

    def gen_data_list(self, x, u):
        return [x, u]


class MIMO:
    """
    p.59 BM-3
    """
    def __init__(self):
        self._pre_x = [0, 0]
        self._pre_u = [-1, -1]

    def get_x(self):
        pass


if __name__ == "__main__":
    data_len = 5000
    file_name = ['./data/train.txt', './data/val.txt']

    sys = SISO()
    for fn in file_name :
        x = 0
        with open(fn, mode='w') as f:
            # sys_info = sys.get_ny_and_nu()
            # f.write(str(sys_info['nx'])+'\n')
            # f.write(str(sys_info['nu'])+'\n')
            for n in range(data_len):
                if n :
                    x = sys.get_x()
                u = sys.get_u()
                s = sys.gen_data_list(x, u)
                sys.set_pre_x_and_u(x, u)

                f.write('\t'.join(map(str, s))+'\n')