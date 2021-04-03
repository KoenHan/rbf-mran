import numpy as np
from copy import deepcopy

from RBF import RBF

class RBF_MRAN:
    def __init__(self, nu, ny, past_sys_input_num, past_sys_output_num, init_h, E1, E2, E3, Nw, Sw):
        self._rbf = RBF(
            nu=nu, ny=ny, init_h=init_h,
            input_size = past_sys_input_num*nu + past_sys_output_num*ny)
        self._rbf_ny = ny
        self._rbf_nu = nu
        self._past_sys_input_num = past_sys_input_num
        self._past_sys_output_num = past_sys_output_num
        self._queue_max_size = ny*past_sys_output_num

        # Step 1で使われるパラメータ
        self._E1 = E1
        self._E2_pow = E2*E2 # ルート取る代わりにしきい値自体を2乗して使う
        self._E3 = E3
        self._Nw = Nw
        self._past_ei_norm_pow = []

        # Step 2で使われるパラメータ
        self._kappa = 0.1 # とりあえずの値

        # Step 4で使われるパラメータ
        #  全部とりあえずの値
        self._p0 = 0.1
        self._P = np.eye(self._rbf.get_param_num())
        self._q = 0.5
        self._R = np.eye(self._rbf_ny, dtype=np.float64) # 観測誤差ノイズ

        # Step 5で使われるパラメータ
        self._delta = 0.1 # とりあえずの値
        self._Sw = Sw
        self._past_o = []
        self._z1 = self._rbf.get_one_unit_param_num()

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
    
    def update_rbf(self, input, yi, debug_cnt):
        f = self._rbf.calc_f(input)
        # todo:
        # 新しいニューロンを追加する前にoを求めるのか
        # 追加したあとに求めるのかを判明する
        o = self._rbf.calc_o()

        # Step 1
        satisfied, ei, di = self._calc_error_criteria(input, f, yi)
        # print("satisfied case ", satisfied)
        z = self._rbf.get_param_num()
        # if debug_cnt%2 == 0 :
        if satisfied == 0 :
            # print("satisfied!!!")
            # Step 2
            self._rbf.add_hidden_unit(ei, input, self._kappa*di)

            # Pの拡張
            zeros = np.zeros((z, self._z1), dtype=np.float64)
            self._P = np.hstack([self._P, zeros])
            tmp = zeros.T
            tmp = np.hstack([tmp, self._p0*np.eye(self._z1)])
            self._P = np.vstack([self._P, tmp])
        else :
            # print("not satisfied!!!")
            # Step 3
            PI = self._rbf.calc_PI()
            # Step 4
            xi = self._rbf.gen_xi()
            K = self._P@PI@np.linalg.inv(self._R + PI.T@self._P@PI)
            xi = xi + K@ei
            self._rbf.update_param_from_xi(xi)

            # Pの更新
            I = np.eye(z)
            self._P = (I - K@PI.T)@self._P + self._q*I
        
        # Step 5
        if o is not None:
            pruned_unit = self._rbf.prune_unit(o, self._Sw, self._delta)
            # Pの調整
            for ui in pruned_unit:
                start = self._rbf_ny + self._z1*ui
                self._P = np.delete(
                    np.delete(self._P, slice(start, start+self._z1), 0),
                    slice(start, start+self._z1), 1)

    def train(self, file_name):
        past_sys_input = [] # 過去のシステム入力
        past_sys_output = [] # 過去のシステム出力
        
        debug_cnt = 0
        with open(file_name, mode='r') as file:
            for line in file:
                data = [float(l) for l in line.split()]

                yi = data[:self._rbf_ny] # 今のシステム出力

                if len(past_sys_output) == self._queue_max_size :
                    # debug_cntは必要なくなったら消していい
                    input = np.array(past_sys_output + past_sys_input, dtype=np.float64)
                    self.update_rbf(input, yi, debug_cnt)

                    for i in range(self._rbf_ny):
                        past_sys_output.pop(0)
                        past_sys_output.append(data[i])
                else :
                    past_sys_output.extend(yi)
                
                past_sys_input = data[-self._rbf_nu:]

                # for debug
                # print(debug_cnt)
                # if debug_cnt == 10:
                #     return
                debug_cnt += 1
    
    def val(self, file_name):
        past_sys_input = [] # 過去のシステム入力
        past_sys_output = [] # 過去のシステム出力

        val_res = []
        
        debug_cnt = 0
        with open(file_name, mode='r') as file:
            for line in file:
                data = [float(l) for l in line.split()]

                yi = data[:self._rbf_ny] # 今のシステム出力

                if len(past_sys_output) == self._queue_max_size :
                    # debug_cntは必要なくなったら消していい
                    input = np.array(past_sys_output + past_sys_input, dtype=np.float64)
                    f = self._rbf.calc_f(input)
                    val_res.append(f)

                    for i in range(self._rbf_ny):
                        past_sys_output.pop(0)
                        past_sys_output.append(data[i])
                else :
                    past_sys_output.extend(yi)
                
                past_sys_input = data[-self._rbf_nu:]

                # for debug
                # print(debug_cnt)
                # if debug_cnt == 10:
                #     return
                debug_cnt += 1
        
        # 結果の保存
        # todo : ちゃんと書く
        with open('./data/val_res.txt', mode='w') as f:
            for res in val_res:
                f.write('\t'.join(map(str, res.tolist()))+'\n')

if __name__ == "__main__" :
    np.set_printoptions(precision=6, suppress=True)
    rbf_mran = RBF_MRAN(
        nu=1, # システム入力(制御入力)の次元
        ny=1, # システム出力ベクトルの次元
        past_sys_input_num=1, # 過去のシステム入力保存数
        past_sys_output_num=3, # 過去のシステム出力保存数
        init_h=0, # スタート時の隠れニューロン数
        E1=0.01, E2=0.01, E3=0.01, Nw=3, Sw=3)

    rbf_mran.train('./data/train.txt')
    rbf_mran.val('./data/val.txt')
    print(rbf_mran._rbf._h)