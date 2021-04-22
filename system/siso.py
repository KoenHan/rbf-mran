import numpy as np
import random

class SISO:
    """
    p.54 BM-1
    """
    def __init__(self, ix=0, iu=-1, ncx=-1, ncu=-1):
        self._init_x = ix
        self._pre_x = ix
        self._pre_u = iu

        # x, uの生成式を切り替える境目
        self._n_change_x = ncx if ncx >= 0 else np.inf
        self._n_change_u = ncu if ncu >= 0 else np.inf

    def get_sys_info(self):
        return {'info_num':2, 'nx':1, 'nu':1}

    def get_x(self, n):
        if n == 0 : return self._init_x
        elif n < self._n_change_x :
            x = 29/40*np.sin((16*self._pre_u + 8*self._pre_x)/(3 + 4*self._pre_u**2 + 4*self._pre_x**2))
            x += 0.2*(self._pre_u + self._pre_x)
        else :
            x = 37/82*np.cos((32*self._pre_u)/(10 + 13*self._pre_u**2 + 6*self._pre_x**2))
            x += 0.4*self._pre_x
        return x

    def get_init_x(self):
        return self._init_x
    
    def get_u(self, n):
        # nは使わないけど整合性のために書いておく
        return random.uniform(-1, 1)

    def set_pre_x_and_u(self, x, u):
        self._pre_x = x
        self._pre_u = u

    def gen_data_list(self, x, u):
        return [x, u]
