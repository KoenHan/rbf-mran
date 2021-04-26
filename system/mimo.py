import numpy as np
import random
# import math

class MIMO:
    """
    p.59 BM-3
    """
    def __init__(self, ix=[0, 0], iu=[-1, -1], ncx=-1, ncu=-1):
        self._init_x = ix
        self._pre_x = [ix]
        self._pre_u = iu
        self._pre_x_limit = 2

        # x, uの生成式を切り替える境目
        self._n_change_x = ncx if ncx >= 0 else np.inf
        self._n_change_u = ncu if ncu >= 0 else np.inf

        self._T = 1000
        self._A_a = None
        self._A_b = None
        self._init_A()

    def get_sys_info(self):
        return {'info_num':2, 'nx':2, 'nu':2}

    def get_x(self, n):
        x = [0, 0]
        if n < 2 : return self._init_x
        # pre_xは後ろから追加されるので先頭がx(i-2)
        elif n < self._n_change_x :
            x[0] = (15*self._pre_u[0]*self._pre_x[0][1])/(2 + 50*self._pre_u[0]**2) 
            x[0] += 0.5*self._pre_u[0] - 0.25*self._pre_x[0][1] + 0.1
            x[1] = (np.sin(np.pi*self._pre_u[1]*self._pre_x[0][0]) + 2*self._pre_u[1])/3
        else :
            p = 2
            if p == 0:
                # 切り替える前と近い例
                x[0] = (30*self._pre_u[0]*self._pre_x[0][1])/(3 + 137*self._pre_u[0]**2) 
                x[0] += 0.23*self._pre_u[0] - 0.75*self._pre_x[0][1] + 0.29
                x[1] = (1.2*np.sin(np.pi*self._pre_u[1]*self._pre_x[0][0]) + 1.5*self._pre_u[1])/5
            elif p == 1: 
                # グラフがギザギザになって学習できてない例
                # mimo_change_jaggedに保存済み．
                x[0] = (3*self._pre_u[0]*self._pre_x[0][1])/(23 + 137*self._pre_u[0]**2) 
                x[0] += 0.2*self._pre_u[0] - 4.75*self._pre_x[0][1] - 2.29
                x[1] = (1.5*np.sin(np.pi*self._pre_u[1]*self._pre_x[0][0]) + 13*self._pre_u[1])/2
                # スケールを切り替え前と近い値にすると学習できるようになる
                # x[0] /= 20
                # x[1] /= 7.5
            elif p == 2:
                # 学習が速くなる例．
                x[0] = (20*self._pre_u[0]*self._pre_x[0][1])/(23 + 15*self._pre_x[0][0]**2) 
                x[0] += 0.25*self._pre_u[0] - 0.75*self._pre_x[0][1] - 0.2
                x[1] = (np.sin(np.pi*self._pre_u[1]*(self._pre_x[0][1]+self._pre_x[0][0])) + 10*self._pre_u[1])/7
            elif p == 3:
                # 学習が大幅に遅くなる+学習できない例（total MAEが167ぐらいになる）
                # mimo_change_failedに保存済み．
                x[0] = (20*self._pre_u[0]*self._pre_x[0][1])/(13 + 15*self._pre_x[0][0]**2) 
                x[0] += 0.25*self._pre_u[0] - 0.75*self._pre_x[0][1] - 0.2
                x[1] = np.sin(np.pi*self._pre_u[1]*self._pre_x[0][1]) + 10*self._pre_u[1]
                x[1] /= 7*self._pre_x[1][0]
            elif p == 4:
                # 切り替えた直後にしばらく振動する例．
                # mimo_change_vabrateに保存済み．
                x[0] = (14*self._pre_u[0]*self._pre_x[0][1])/(36 + 17*self._pre_u[0]**2) 
                x[0] += 0.23*self._pre_u[0] + 0.75*self._pre_x[0][1] - 0.29
                x[1] = (np.cos(np.pi*self._pre_u[1]*self._pre_x[0][0]*self._pre_x[0][1]) + 2*self._pre_x[1][0])/3
            elif p == 5:
                # xを切り替え前の2倍にしただけだが不安定になる例．（0.5倍は学習できるので特に保存してない．）
                # mimo_change_x2に保存済み．
                x[0] = (15*self._pre_u[0]*self._pre_x[0][1])/(2 + 50*self._pre_u[0]**2) 
                x[0] += 0.5*self._pre_u[0] - 0.25*self._pre_x[0][1] + 0.1
                x[1] = (np.sin(np.pi*self._pre_u[1]*self._pre_x[0][0]) + 2*self._pre_u[1])/3
                x[0] *= 2
                x[1] *= 2
        return x
    
    def get_init_x(self):
        return self._init_x
    
    def get_u(self, n):
        # p.60に合わせた
        # return [random.uniform(-1, 1), random.uniform(-1, 1)]
        nn = n%(self._T/2)
        if nn == 0 :
            self._init_A()
        omega = 2*np.pi/self._T
        A0 = 2*(self._A_max[0] - self._A_min[0])/self._T*nn + self._A_min[0]
        u0 = A0*np.sin(20*omega*n + 12*np.sin(omega*(n - self._T/2)))
        if abs(u0) < 1e-6: u0 = 0.0
        A1 = 2*(self._A_min[1] - self._A_max[1])/self._T*nn + self._A_max[1]
        u1 = A1*np.sin(10*omega*n + 6*np.sin(omega*(n)))
        if abs(u1) < 1e-6: u1 = 0.0
        return [u0, u1]
        # memo : tmp1:tmp2=10:6の割合を保ちつつ係数の大きさを調整するとうまく行く
        # A*np.sin(tmp1*np.deg2rad(n) + tmp2*np.sin(np.deg2rad(n)))

    def _init_A(self):
        self._A_min = [random.uniform(0.2, 0.3), random.uniform(0.2, 0.3)]
        self._A_max = [1, 1]
    
    def set_pre_x_and_u(self, x, u):
        if len(self._pre_x) == self._pre_x_limit :
            del self._pre_x[0]
        self._pre_x.append(x)
        self._pre_u = u
    
    def gen_data_list(self, x, u):
        return x + u

if __name__=="__main__":
    import matplotlib.pyplot as plt
    mimo = MIMO()
    ul = []
    limit = 1000
    x = [i for i in range(limit)]
    for n in x:
        ul.append(mimo.get_u(n))
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(211)
    ax.plot(x, ul)
    plt.grid()
    plt.show()
