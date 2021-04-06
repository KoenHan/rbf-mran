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
        self._kappa = 1.0 # とりあえずの値

        # Step 4で使われるパラメータ
        #  全部とりあえずの値
        self._p0 = 1.0
        self._P = np.eye(self._rbf.get_param_num())
        self._q = 0.1
        self._R = np.eye(self._rbf_ny, dtype=np.float64) # 観測誤差ノイズ

        # Step 5で使われるパラメータ
        self._delta = 0.0001 # p.55
        self._Sw = Sw
        self._past_o = []
        self._z1 = self._rbf.get_one_unit_param_num()

        # p.55の実験に必要
        self._eps_max = 1.2
        self._eps_min = 0.6
        self._gamma = 0.997
        self._gamma_n = 1

        self._ei_abs_queue = [] # 学習中の誤差計算
        self._Id_hist = [] # 学習中の誤差履歴
        self._h_hist = [init_h] # 学習中の隠れニューロン数履歴

    def calc_E3(self):
        """
        p.38の実験のE3の計算
        """
        self._gamma_n *= self._gamma
        return max(self._eps_max*self._gamma_n, self._eps_min)

    def _calc_error_criteria(self, input, f, yi):
        """
        Step 1の実装
        Returns: 
            -(bool):
                3つの基準値を満たしているかどうか．
                満たしているならTrue，そうでないならFalse．
            ei(ndarray(np.float64)):
                式3.4のei
            ei_norm:

            myu_ir(ndarray(np.float64)):←いらないので消した
                式3.6のmyu_ir
        """
        ei = yi - f
        ei_norm = np.linalg.norm(ei, ord=2)
        di = self._rbf.get_closest_unit_myu_and_dist(input)

        self._past_ei_norm_pow.append(ei@ei)
        if len(self._past_ei_norm_pow) > self._Nw :
            del self._past_ei_norm_pow[0]

        case = 0
        if ei_norm <= self._E1 :
            case = 1
        elif sum(self._past_ei_norm_pow) <= self._E2_pow*self._Nw :
            case = 2
        # elif di <= self._E3 :
        elif di <= self.calc_E3(): # p.55の実験に合わせた
            case = 3

        return case, ei, ei_norm, di
    
    def update_rbf(self, input, yi, cnt):
        f = self._rbf.calc_f(input)
        o = self._rbf.calc_o()

        # Step 1
        satisfied, ei, ei_norm, di = self._calc_error_criteria(input, f, yi)
        # print("satisfied case ", satisfied)
        z = self._rbf.get_param_num()
        # if cnt%2 == 0 :
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

        # 学習中の誤差（式3.16）の算出及び保存
        self._ei_abs_queue.append(ei_norm)
        if len(self._ei_abs_queue) > self._Nw:
            del self._ei_abs_queue[0]
            self._Id_hist.append(sum(self._ei_abs_queue)/self._Nw)
        # 学習中の隠れニューロン数保存
        self._h_hist.append(self._rbf.get_h())

        # todo : ネットワークパラメータを何かのファイルに保存

    def train(self, file_name):
        past_sys_input = [] # 過去のシステム入力
        past_sys_output = [] # 過去のシステム出力
        
        cnt = 0
        with open(file_name, mode='r') as file:
            for line in file:
                data = [float(l) for l in line.split()]

                yi = data[:self._rbf_ny] # 今のシステム出力

                if len(past_sys_output) == self._queue_max_size :
                    input = np.array(past_sys_output + past_sys_input, dtype=np.float64)
                    self.update_rbf(input, yi, cnt)

                    for i in range(self._rbf_ny):
                        del past_sys_output[0]
                        past_sys_output.append(data[i])
                else :
                    past_sys_output.extend(yi)
                
                past_sys_input = data[-self._rbf_nu:]

                cnt += 1

        # 誤差履歴，隠れニューロン数履歴の保存
        with open('./model/history/error.txt', mode='w') as f:
            f.write(str(self._Nw)+'\n')
            f.write('\n'.join(map(str, self._Id_hist))+'\n')
        with open('./model/history/h.txt', mode='w') as f:
            f.write('\n'.join(map(str, self._h_hist))+'\n')
    
    def val(self, file_name):
        past_sys_input = [] # 過去のシステム入力
        past_sys_output = [] # 過去のシステム出力

        val_res = []
        
        cnt = 0
        with open(file_name, mode='r') as file:
            for line in file:
                data = [float(l) for l in line.split()]

                yi = data[:self._rbf_ny] # 今のシステム出力

                if len(past_sys_output) == self._queue_max_size :
                    # cntは必要なくなったら消していい
                    input = np.array(past_sys_output + past_sys_input, dtype=np.float64)
                    f = self._rbf.calc_f(input)
                    val_res.append(f)

                    for i in range(self._rbf_ny):
                        del past_sys_output[0]
                        past_sys_output.append(data[i])
                else :
                    past_sys_output.extend(yi)
                
                past_sys_input = data[-self._rbf_nu:]

                cnt += 1
        
        # 結果の保存
        # memo : ちゃんと書きたい
        with open('./data/pre_res.txt', mode='w') as f:
            for res in val_res:
                f.write('\t'.join(map(str, res.tolist()))+'\n')

if __name__ == "__main__" :
    np.set_printoptions(precision=6, suppress=True)
    rbf_mran = RBF_MRAN(
        nu=1, # システム入力(制御入力)の次元
        ny=1, # システム出力ベクトルの次元
        past_sys_input_num=1, # 過去のシステム入力保存数
        past_sys_output_num=1, # 過去のシステム出力保存数
        init_h=0, # スタート時の隠れニューロン数
        E1=0.01, E2=0.01, E3=1.2, Nw=48, Sw=48)

    rbf_mran.train('./data/train.txt')
    rbf_mran.val('./data/val.txt')