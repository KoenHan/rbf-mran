import numpy as np
import random

class MIMO:
    """
    p.59 BM-3
    """
    def __init__(self, ix=[0, 0], iu=[-1, -1]):
        self._init_x = ix
        self._pre_x = [ix]
        self._pre_u = iu
        self._pre_x_limit = 2

    def get_sys_info(self):
        return {'info_num':2, 'nx':2, 'nu':2}

    def get_x(self, n):
        x = [0, 0]
        if n < 2 : return self._init_x
        # pre_xは後ろから追加されるので先頭がx(i-2)
        x[0] = (15*self._pre_u[0]*self._pre_x[0][1])/(2 + 50*self._pre_u[0]**2) 
        x[0] += 0.5*self._pre_u[0] - 0.25*self._pre_x[0][1] + 0.1
        x[1] = (np.sin(np.pi*self._pre_u[1]*self._pre_x[0][0]) + 2*self._pre_u[1])/3
        return x
    
    def get_init_x(self):
        return self._init_x
    
    def get_u(self):
        return [random.uniform(-1, 1), random.uniform(-1, 1)]
    
    def set_pre_x_and_u(self, x, u):
        if len(self._pre_x) == self._pre_x_limit :
            del self._pre_x[0]
        self._pre_x.append(x)
        self._pre_u = u
    
    def gen_data_list(self, x, u):
        return x + u
