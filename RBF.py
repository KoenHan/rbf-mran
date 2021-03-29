import numpy as np
from copy import deepcopy

class RBF:
    def __init__(self, ny, h, input_size):
        self._ny = ny
        self._h = h
        self._input_size = input_size

        # ネットワークパラメータ
        self._sigma = np.array([10 for _ in range(self._h)], dtype=np.float64)
        # self._sigma2 = np.array([10 for _ in range(self._h)], dtype=np.float64)
        self._myu = np.array([0 for _ in range(self._input_size*self._h)], dtype=np.float64).reshape(self._input_size, self._h)
        self._w0 = np.array([1 for _ in range(self._ny)], dtype=np.float64)
        self._wk = np.array([k for k in range(self._ny*self._h)], dtype=np.float64).reshape(self._ny, self._h)

        self._r = np.empty(self._h)
        self._r2 = np.empty(self._h)
        self._phi = np.empty(self._h)

    def _calc_diff_from_myu(self, myu, xi):
        return xi - myu

    def _calc_pow(self, r):
        return r@r

    def update_param_from_xi(self, xi):
        # np.set_printoptions(precision=10, suppress=True)
        self._w0 = xi[:self._ny]
        z1 = self.get_one_unit_param_num()
        left = self._ny
        for hi in range(self._h):
            self._wk[:, hi] = xi[left:left+self._ny]
            left += self._ny
            self._myu[:, hi] = xi[left:left+self._input_size]
            left += self._input_size
            self._sigma[hi] = xi[left]
            left += 1
        return

    def gen_xi(self):
        xi = deepcopy(self._w0)

        for hi in range(self._h):
            xi = np.hstack([xi, self._wk[:, hi], self._myu[:, hi], self._sigma[hi]])

        return xi
    
    def get_param_num(self):
        """
        更新できるパラメータの数 = count(w0) + count(wk) + count(myu) + count(sigma)
        """
        return self._w0.size + self._wk.size + self._myu.size + self._sigma.size

    def get_one_unit_param_num(self):
        """
        一つのニューロンが持ってるパラメータの数 = ny + input_size + 1(sigma)
        """
        return self._ny + self._input_size + 1

    def get_closest_unit_myu_and_dist(self, xi):
        """
        xiに一番近いμとそのμまでの距離を求める
        Args:
            xi(np.array(np.float64)):
                対象ベクトル
        Returns:
            myu_ir(np.array(np.float64)):
                xiに一番近いμ
            di(double):
                xiと一番近いμとの距離
        """
        di = np.inf
        for col in range(self._h):
            e = np.linalg.norm(xi - self._myu[:, col])
            if e < di:
                di = deepcopy(e)
                myu_ir = deepcopy(self._myu[:, col])
        return myu_ir, di

    def add_hidden_unit(self, weight, myu, sigma):
        # print("before")
        # print(self._wk)
        # print(self._wk.shape)
        # print(self._myu)
        # print(self._myu.shape)
        # print(self._sigma)
        # print(self._sigma.shape)
        self._wk = np.hstack([self._wk, weight.reshape(-1, 1)])
        self._myu = np.hstack([self._myu, myu.reshape(-1, 1)])
        self._sigma = np.append(self._sigma, sigma)
        self._h += 1
        # self._sigma2 = np.append(self._sigma, sigma*sigma)
        # print("after")
        # print(self._wk)
        # print(self._wk.shape)
        # print(self._myu)
        # print(self._myu.shape)
        # print(self._sigma)
        # print(self._sigma.shape)
        return

    def calc_PI(self):
        """
        MRANのStep 3で使われるΠの計算

        Returns:
            -(ndarray(np.float64)):
                式3.8のΠ
        """
        I = np.eye(self._ny, dtype=np.float64)
        PI = deepcopy(I)

        for hi in range(self._h):
            PI = np.hstack([PI, self._phi[hi]*I])
            tmp_a = (2*self._wk[:, hi]/self._sigma[hi]*self._sigma[hi]).reshape(-1, 1)
            tmp_b = self._r[:, hi].reshape(1, -1)
            PI = np.hstack([PI, self._phi[hi]*tmp_a@tmp_b])
            tmp_a /= self._sigma[hi]
            PI = np.hstack([PI, self._phi[hi]*tmp_a*self._r2[hi]])

        return PI.T

    def calc(self, input):
        # 制御入力 u: (1, 1) = (1, nu)
        # システム出力 y: (1, 2) = (1, ny)
        # RBF入力ベクトル x: y*3 + u*1 = (1, 7) = (1, nx)
        # RBF出力ベクトル f: y*1 = (1, 2) = (1, ny)
        # 隠れニューロン数 h = 5
        # バイアス w0: f*1 = (1, ny)
        # 重み wk: h*ny = (5, 2)
        # 平均 μ: (1, h) = (1, 5)
        # 分散 σ^2: (1, h) = (1, 5)

        print("input ",input.shape)
        print("myu ", self._myu.shape)
        # r, r2, phiは後（update_param）で使うので保存
        self._r = np.apply_along_axis(self._calc_diff_from_myu, 0, self._myu, input)
        self._r2 = np.apply_along_axis(self._calc_pow, 0, self._r)
        print("r", self._r.shape)
        print("r2", self._r2.shape)
        self._phi = np.exp(-self._r2/(self._sigma*self._sigma))
        print("phi ",self._phi.shape)
        print("w0 ", self._w0.shape)
        print("wk ", self._wk.shape)
        f = self._w0 + self._wk@self._phi
        print("f ", f.shape)
        # a = np.array([5, 6, 7, 8, 9])
        # b = np.array([0.92219369, 0.92219369, 0.92219369, 0.92219369, 0.92219369])
        # print(a@b)
        return f

if __name__ == "__main__" :
    print('nothing to do')
