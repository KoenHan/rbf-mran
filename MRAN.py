import numpy as np
from copy import deepcopy

from RBF import RBF

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

        # Step 4で使われるパラメータ
        self._p0 = 0.1
        self._P = np.eye(self._rbf.get_param_num())
        self._q = 0.5

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
        myu_ir, di = self._rbf.get_closest_unit_myu_and_dist(input)

        self._past_ei_norm_pow.append(ei@ei)
        if len(self._past_ei_norm_pow) > self._Nw :
            self._past_ei_norm_pow.pop(0)

        if np.linalg.norm(ei, ord=2) <= self._E1 :
            return (False, ei, myu_ir)
        elif sum(self._past_ei_norm_pow) <= self._E2_pow*self._Nw :
            return (False, ei, myu_ir)
        elif di <= self._E3 :
            return (False, ei, myu_ir)

        return (True, None, None)
    
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
        z = self._rbf.get_param_num()
        z1 = self._rbf.get_one_unit_param_num()
        if True :
        # if satisfied :
            # Step 2
            self._add_new_rbf_hidden_unit(ei, input, myu_ir)

            # Pの拡張
            zeros = np.zeros((z, z1), dtype=np.float64)
            self._P = np.hstack([self._P, zeros])
            tmp = zeros.T
            tmp = np.hstack([tmp, self._p0*np.eye(z1)])
            self._P = np.vstack([self._P, tmp])
        else :
            # Step 3
            PI = self._rbf.calc_PI()
            # Step 4
            xi = self._rbf.gen_xi()
            # todo : Rの計算方法
            R = np.eye(2, dtype=np.float64)
            K = self._P@PI@np.linalg.inv(R + PI.T@self._P@PI)
            xi = xi + K@ei
            self._rbf.update_param_from_xi(xi)

            # Pの更新
            I = np.eye(z)
            self._P = (I - K@PI.T)@self._P + self._q*I
            print(self._P.shape)
        
        # Step 5

        return

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