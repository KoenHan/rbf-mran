import numpy as np
from copy import deepcopy

class Unit:
    """
    RBFの隠れ層のニューロン
    各ニューロンに関連するパラメータをただの行列として保持すると，
    MRANのStep 5でニューロンを消す時にめんどくさくなるのでこのクラスは必要
    """
    def __init__(self, wk, myu, sigma):
        """
        このクラスのメンバー変数はprivate扱いしない
        """
        self.wk = wk
        self.myu = myu
        self.sigma = sigma

        self.past_o = []

class RBF:
    """
    ネットワークパラメータはニューロンごとで管理し，計算時に行列を構築する
    """
    def __init__(self, nu, ny, init_h, input_size):
        self._nu = nu
        self._ny = ny
        self._h = init_h
        self._input_size = input_size

        # ネットワークパラメータ
        self._w0 = np.array([0 for _ in range(self._ny)], dtype=np.float64) # 隠れニューロン0から始まるからバイアスはニューロンのパラメータではない
        self._wk = None # (ny, h)
        self._myu = None # (input_size, h)
        self._sigma = None
        
        self._hidden_unit = [] # 隠れニューロン保存用
        if self._h : # 基本ここで作成しないけど一応書いておく
            for hi in range(self._h):
                self._hidden_unit.append(Unit(
                    wk = np.array([k for k in range(self._ny)], dtype=np.float64),
                    myu = np.array([0 for _ in range(self._input_size)], dtype=np.float64),
                    sigma = 10 + hi
                ))
            self._gen_network_from_hidden_unit()

        # あとで使うための宣言
        self._r = None
        self._r2 = None
        self._phi = None

        # 隠れニューロン数履歴保存用
        self._h_hist = []

    def get_h(self):
        return self._h
    
    def _gen_network_from_hidden_unit(self):
        wk = []
        myu = []
        sigma = []
        for unit in self._hidden_unit:
            wk.append(unit.wk)
            myu.append(unit.myu)
            sigma.append(unit.sigma)
        self._wk = np.vstack(wk).T
        self._myu = np.vstack(myu).T
        self._sigma = np.array(sigma, np.float64)

    def gen_xi(self):
        xi = deepcopy(self._w0)

        for hi in range(self._h):
            xi = np.hstack([xi, self._wk[:, hi], self._myu[:, hi], self._sigma[hi]])

        return xi

    def update_param_from_xi(self, xi):
        self._w0 = xi[:self._ny]
        left = self._ny
        for unit in self._hidden_unit:
            unit.wk = xi[left:left+self._ny]
            left += self._ny
            unit.myu = xi[left:left+self._input_size]
            left += self._input_size
            unit.sigma = xi[left]
            left += 1
    
    def get_param_num(self):
        """
        更新できるパラメータの数 = count(w0) + count(wk) + count(myu) + count(sigma)
        """
        if self._h == 0:
            return self._ny
        return self._ny + self._wk.size + self._myu.size + self._sigma.size

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
            myu_ir(np.array(np.float64)):←消した
                xiに一番近いμ
            di(double):
                xiと一番近いμとの距離
        """
        di = 1.0e8
        for col in range(self._h):
            e = np.linalg.norm(xi - self._myu[:, col])
            if e < di:
                di = deepcopy(e)
        return di

    def add_hidden_unit(self, weight, myu, sigma):
        self._hidden_unit.append(Unit(
            wk = weight,
            myu = myu,
            sigma = sigma
        ))
        self._h += 1
    
    def prune_unit(self, o, Sw, delta):
        """
        隠れニューロンの削除
        """
        def must_prune(past_o):
            for p_o in past_o:
                if np.all(p_o >= delta) :
                    return False
            return True

        # 削除すべきニューロンのindexの列挙
        must_prune_unit = []
        for hi in range(o.shape[1]) :
            unit = self._hidden_unit[hi]
            unit.past_o.append(o[:, hi])
            if len(unit.past_o) > Sw :
                del unit.past_o[0]
            if must_prune(unit.past_o) :
                must_prune_unit.append(hi)

        # 削除すべきニューロンの削除
        must_prune_unit.reverse()
        for ui in must_prune_unit:
            del self._hidden_unit[ui]
            self._h -= 1

        return must_prune_unit

    def calc_PI(self):
        """
        MRANのStep 3で使われるΠの計算
        """
        I = np.eye(self._ny, dtype=np.float64)
        PI = deepcopy(I)

        for hi in range(self._h):
            PI = np.hstack([PI, self._phi[hi]*I])
            tmp_a = (2*self._wk[:, hi]/(self._sigma[hi]*self._sigma[hi])).reshape(-1, 1)
            tmp_b = self._r[:, hi].reshape(1, -1)
            PI = np.hstack([PI, self._phi[hi]*tmp_a@tmp_b])
            tmp_a /= self._sigma[hi]
            PI = np.hstack([PI, self._phi[hi]*tmp_a*self._r2[hi]])
        return PI.T

    def calc_f(self, input):
        if self._h == 0:
            return self._w0

        self._gen_network_from_hidden_unit()

        # r, r2, phiは後（update_param）で使うので保存
        # self._r = np.apply_along_axis(lambda myu, xi: xi - myu, 0, self._myu, input)
        self._r = np.tile(input, (self._h, 1)).T - self._myu # こっちのほうが速い
        self._r2 = np.apply_along_axis(lambda a: a@a, 0, self._r)
        self._phi = np.exp(-self._r2/(self._sigma*self._sigma))
        return self._w0 + self._wk@self._phi
    
    def calc_o(self):
        """
        MRANのStep 5で使われるoの計算
        """
        if self._h == 0:
            return None
        o = self._wk*np.tile(self._phi, (self._ny, 1)) # こっちのほうが若干速い
        # o = deepcopy(self._wk)
        # for hi in range(self._h):
        #     o[:, hi] *= self._phi[hi]
        return o/o.max()

