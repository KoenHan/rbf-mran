import numpy as np
from copy import deepcopy

class MRAN:
    def __init__(self, rbf, E1, E2, E3, Nw):
        self._rbf = rbf

        # Step 1で使われるパラメータ
        self._E1 = E1
        self._E2_pow = E2*E2 # ルート取る代わりにしきい値自体を2乗して使う
        self._E3 = E3
        self._Nw = Nw
        self._past_ei_norm_pow = []

        # Step 2で使われるパラメータ
        self._kappa = 0.1

    def _calc_error_criteria(self, input, f, yi):
        """
        Step 1の実装
        Returns: 
            -(bool):
                3つの基準値を満たしているかどうか．
                満たしているならTrue，そうでないならFalse．
            ei(ndarray(np.float64)):
                式3.4のei
            myu_ir(ndarray(np.float64)):
                式3.6のmyu_ir
        """
        ei = yi - f
        self._past_ei_norm_pow.append(ei@ei)
        if len(self._past_ei_norm_pow) > self._Nw :
            self._past_ei_norm_pow.pop(0)

        if np.linalg.norm(ei, ord=2) <= self._E1 :
            return (False, None, None)
        elif sum(self._past_ei_norm_pow) <= self._E2_pow*self._Nw :
            return (False, None, None)

        myu_ir, di = self._rbf.get_closest_unit_myu_and_dist(input)
        # if di <= self._E3 :
        #     return (False, None, None)

        return (True, ei, myu_ir)
    
    def _add_new_rbf_hidden_unit(self, ei, input, myu_ir):
        sigma = self._kappa*np.linalg.norm(input - myu_ir)
        self._rbf.add_hidden_unit(ei, myu_ir, sigma)
        return
    
    def update_rbf(self, input1, input2, yi):
        input = np.array(input1, dtype=np.float64)
        input = np.append(input, input2)
        # input_size = input.shape[0]

        f = self._rbf.calc(input)

        # Step 1
        satisfied, ei, myu_ir = self._calc_error_criteria(input, f, yi)
        if satisfied :
            # Step 2
            print('goto step 2')
            self._add_new_rbf_hidden_unit(ei, input, myu_ir)
        else :
            # Step 3
            # Step 4
            print('goto step 3, 4')
            pass
        
        # Step 5

        return

class RBF:
    def __init__(self, ny, h, input_size):
        self._ny = ny
        self._h = h
        self._input_size = input_size

        # ネットワークパラメータ
        self._sigma2 = np.array([10 for _ in range(self._h)], dtype=np.float64)
        self._myu = np.array([0 for _ in range(self._h*self._input_size)], dtype=np.float64).reshape(self._input_size, self._h)
        self._w0 = np.array([1 for _ in range(self._ny)], dtype=np.float64)
        self._wk = np.array([k for k in range(self._ny*self._h)], dtype=np.float64).reshape(self._ny, self._h)

    def _calc_norm_from_myu(self, xi, myu):
        return np.linalg.norm(xi - myu, ord=2)

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
        # print(self._sigma2)
        # print(self._sigma2.shape)
        self._wk = np.append(self._wk, weight.reshape(-1, 1), axis=1)
        self._myu = np.append(self._myu, myu.reshape(-1, 1), axis=1)
        self._sigma2 = np.append(self._sigma2, sigma*sigma)
        # print("after")
        # print(self._wk)
        # print(self._wk.shape)
        # print(self._myu)
        # print(self._myu.shape)
        # print(self._sigma2)
        # print(self._sigma2.shape)
        return

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
        r2 = np.apply_along_axis(self._calc_norm_from_myu, 0, self._myu, input)
        r2 *= r2
        phi = np.exp(-r2/self._sigma2)#.reshape(1, -1)
        print("phi ",phi.shape)
        print("w0 ", self._w0.shape)
        print("wk ", self._wk.shape)
        f = self._w0 + self._wk@phi
        print(f.shape)
        # a = np.array([5, 6, 7, 8, 9])
        # b = np.array([0.92219369, 0.92219369, 0.92219369, 0.92219369, 0.92219369])
        # print(a@b)
        return f

def main():
    nu = 1 # システム入力(制御入力)の次元
    ny = 2 # システム出力ベクトルの次元
    past_sys_input_num = 1# 過去のシステム入力保存数
    past_sys_output_num = 3 # 過去のシステム出力保存数

    queue_max_size = ny*past_sys_output_num
    past_sys_output = [] # 過去のシステム出力
    
    h = 5 # 隠れニューロン数
    rbf = RBF(
        ny, h,
        input_size = past_sys_input_num*nu + past_sys_output_num*ny)
    mran = MRAN(rbf, E1 = 1, E2 = 2, E3 = 3, Nw = 3)

    debug_cnt = 0
    with open('./data/data.txt', mode='r') as file:
        for line in file:
            data = tuple(float(l) for l in line.split())

            yi = data[0:ny] # 今のシステム出力
            
            # for num in past_sys_output:
            #     print('o: '+str(num))
            # print('i: '+str(past_sys_input))

            if len(past_sys_output) == queue_max_size :
                # rbf.calc(past_sys_output, past_sys_input)
                mran.update_rbf(past_sys_output, past_sys_input, yi)

                for i in range(ny):
                    past_sys_output.pop(0)
                    past_sys_output.append(data[i])
            else :
                past_sys_output.extend(yi)
            
            past_sys_input = data[-nu:]

            # for debug
            print(debug_cnt)
            if debug_cnt == 3 :
                return
            debug_cnt += 1
    return

if __name__ == "__main__" :
    main()