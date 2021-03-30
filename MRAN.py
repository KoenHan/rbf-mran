import numpy as np
from copy import deepcopy

from RBF import RBF

class MRAN:
    def __init__(self, rbf, E1, E2, E3, Nw, Sw):
        self._rbf = rbf

        # Step 1で使われるパラメータ
        self._E1 = E1
        self._E2_pow = E2*E2 # ルート取る代わりにしきい値自体を2乗して使う
        self._E3 = E3
        self._Nw = Nw
        self._past_ei_norm_pow = []

        # Step 2で使われるパラメータ
        self._kappa = 0.1 # とりあえずの値

        # Step 4で使われるパラメータ
        self._p0 = 0.1 # とりあえずの値
        self._P = np.eye(self._rbf.get_param_num()) # とりあえずの値
        self._q = 0.5 # とりあえずの値

        # Step 5で使われるパラメータ
        self._delta = 0.1 # とりあえずの値
        self._Sw = Sw
        self._past_o = []

    def _calc_error_criteria(self, input, f, yi):
        """
        Step 1の実装
        Returns: 
            -(bool):
                3つの基準値を満たしているかどうか．
                満たしているならTrue，そうでないならFalse．
            ei(ndarray(np.float64)):
                式3.4のei
            myu_ir(ndarray(np.float64)):←いらないので消した
                式3.6のmyu_ir
        """
        ei = yi - f
        di = self._rbf.get_closest_unit_myu_and_dist(input)
        # myu_ir, di = self._rbf.get_closest_unit_myu_and_dist(input)

        self._past_ei_norm_pow.append(ei@ei)
        if len(self._past_ei_norm_pow) > self._Nw :
            self._past_ei_norm_pow.pop(0)

        case = 0
        if np.linalg.norm(ei, ord=2) <= self._E1 :
            case = 1
        elif sum(self._past_ei_norm_pow) <= self._E2_pow*self._Nw :
            case = 2
        elif di <= self._E3 :
            case = 3

        return case, ei, di

    def _listup_must_prune_unit(self):
        # print("past o ", self._past_o)
        # print("past o ", np.concatenate(self._past_o, dtype=np.float64))
        # todo : 実装
        return
    
    def update_rbf(self, input1, input2, yi, debug_cnt):
        input = np.array(input1 + input2, dtype=np.float64)

        f, o = self._rbf.calc_output(input)
        # memo : 多分ここでやる必要がなくなる？
        # self._past_o.append(o)
        # if len(self._past_o) > self._Sw :
        #     self._past_o.pop(0)

        # Step 1
        satisfied, ei, di = self._calc_error_criteria(input, f, yi)
        print("satisfied case ", satisfied)
        z = self._rbf.get_param_num()
        z1 = self._rbf.get_one_unit_param_num()
        # if debug_cnt%2 == 0 :
        if satisfied == 0 :
            print("satisfied!!!")
            # Step 2
            self._rbf.add_hidden_unit(ei, input, self._kappa*di)

            # Pの拡張
            zeros = np.zeros((z, z1), dtype=np.float64)
            print("z ", z)
            print(self._P.shape)
            print(zeros.shape)
            self._P = np.hstack([self._P, zeros])
            tmp = zeros.T
            tmp = np.hstack([tmp, self._p0*np.eye(z1)])
            self._P = np.vstack([self._P, tmp])
            print(self._P.shape)
        else :
            print("not satisfied!!!")
            # Step 3
            PI = self._rbf.calc_PI()
            # Step 4
            xi = self._rbf.gen_xi()
            # todo : Rの計算方法
            R = np.eye(2, dtype=np.float64)
            K = self._P@PI@np.linalg.inv(R + PI.T@self._P@PI)
            xi = xi + K@ei
            self._rbf.update_param_from_xi(xi)
            print("xi ", xi)

            # Pの更新
            I = np.eye(z)
            self._P = (I - K@PI.T)@self._P + self._q*I
        
        # Step 5
        prune_unit_id = self._listup_must_prune_unit()

        return

def main():
    nu = 1 # システム入力(制御入力)の次元
    ny = 2 # システム出力ベクトルの次元
    past_sys_input_num = 1# 過去のシステム入力保存数
    past_sys_output_num = 3 # 過去のシステム出力保存数

    queue_max_size = ny*past_sys_output_num
    past_sys_output = [] # 過去のシステム出力
    
    h = 0 # 隠れニューロン数
    rbf = RBF(
        ny, h,
        input_size = past_sys_input_num*nu + past_sys_output_num*ny)
    mran = MRAN(rbf, E1 = 0.01, E2 = 0.01, E3 = 0.01, Nw = 3, Sw = 3)

    debug_cnt = 0
    with open('./data/data.txt', mode='r') as file:
        for line in file:
            data = [float(l) for l in line.split()]

            yi = data[0:ny] # 今のシステム出力
            
            # for num in past_sys_output:
            #     print('o: '+str(num))
            # print('i: '+str(past_sys_input))

            if len(past_sys_output) == queue_max_size :
                # rbf.calc(past_sys_output, past_sys_input)
                mran.update_rbf(past_sys_output, past_sys_input, yi, debug_cnt)

                for i in range(ny):
                    past_sys_output.pop(0)
                    past_sys_output.append(data[i])
            else :
                past_sys_output.extend(yi)
            
            past_sys_input = data[-nu:]

            # for debug
            print(debug_cnt)
            if debug_cnt == 10:
                return
            debug_cnt += 1
    return

if __name__ == "__main__" :
    np.set_printoptions(precision=10, suppress=True)
    main()