import numpy as np
import time
import random
from copy import deepcopy

from RBF import RBF

class RBF_MRAN:
    def __init__(self, nu, ny, past_sys_input_num, past_sys_output_num,
            init_h, E1, E2, E3, E3_max, E3_min, gamma, Nw, Sw, kappa=1.0,
            p0=1, q=0.1, realtime=False, input_delay=0, output_delay=0):
        self._rbf = RBF(
            nu=nu, ny=ny, init_h=init_h,
            input_size = past_sys_input_num*nu + past_sys_output_num*ny)
        self._rbf_ny = ny
        self._rbf_nu = nu
        self._past_sys_input = [] # 過去のシステム入力
        self._past_sys_input_num = past_sys_input_num
        self._past_sys_input_size = nu*past_sys_input_num
        self._past_sys_input_limit = self._past_sys_input_size + nu*input_delay
        self._past_sys_output = [] # 過去のシステム出力
        self._past_sys_output_num = past_sys_output_num
        self._past_sys_output_size = ny*past_sys_output_num
        self._past_sys_output_limit = self._past_sys_output_size + ny*output_delay

        # Step 1で使われるパラメータ
        self._E1 = float(E1)
        self._E2_pow = float(E2)*float(E2) # ルート取る代わりにしきい値自体を2乗して使う
        self._E3 = float(E3)
        self._E3_max = float(E3_max)
        self._E3_min = float(E3_min)
        self._gamma = float(gamma)
        self._Nw = Nw
        self._past_ei_norm_pow = []

        # Step 2で使われるパラメータ
        self._kappa = kappa

        # Step 4で使われるパラメータ
        #  全部とりあえずの値
        self._p0 = p0
        self._P = np.eye(self._rbf.get_param_num())
        self._q = q # 小さくする
        # self._q = 0.1
        self._R = 0.001*np.eye(self._rbf_ny, dtype=np.float64) # 観測誤差ノイズ 大きくする

        # Step 5で使われるパラメータ
        self._delta = 0.0001 # p.55
        self._Sw = Sw
        self._past_o = []
        self._z1 = self._rbf.get_one_unit_param_num()

        # p.55の実験に必要
        self._gamma_n = 1

        self._ei_abs = [] # 学習中の誤差履歴(MAE)
        self._Id_hist = [] # 学習中の誤差履歴(式3.16)
        self._h_hist = [init_h] # 学習中の隠れニューロン数履歴
        self._train_pre_res = [] # リアルタイムのシステム同定の結果の保存
        self._test_pre_res = [] # 検証時の予測結果の保存

        self._update_rbf_time = [] # 時間計測
        self.realtime = realtime # リアルタイムのシステム同定の場合のフラグ

    def _calc_E3(self):
        # return self._E3
        """
        p.55の実験のE3の計算
        """
        self._gamma_n *= self._gamma
        return max(self._E3_max*self._gamma_n, self._E3_min)

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
        elif di <= self._calc_E3(): # p.55の実験に合わせた
            case = 3

        return case, ei, ei_norm, di

    def update_rbf(self, input, yi):
        f = self._rbf.calc_f(input)
        o = self._rbf.calc_o()

        # Step 1
        satisfied, ei, ei_norm, di = self._calc_error_criteria(input, f, yi)
        # print("satisfied case ", satisfied)
        z = self._rbf.get_param_num()
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
            chi = self._rbf.gen_chi()
            K = self._P@PI@np.linalg.inv(self._R + PI.T@self._P@PI)
            chi = chi + K@ei
            self._rbf.update_param_from_chi(chi)

            # Pの更新
            I = np.eye(z)
            self._P = (I - K@PI.T)@self._P + self._q*I
            # self._P = (I - K@PI.T)@self._P + random.uniform(0, 0.1)*I # todo: 勾配方向のランダムステップの実装

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
        self._ei_abs.append(ei_norm)
        if len(self._ei_abs) >= self._Nw:
            self._Id_hist.append(sum(self._ei_abs[-self._Nw:])/self._Nw)
        # 学習中の隠れニューロン数保存
        self._h_hist.append(self._rbf.get_h())

        return f

    def _gen_input(self):
        return np.array(self._past_sys_output[:self._past_sys_output_size]
            + self._past_sys_input[:self._past_sys_input_size], dtype=np.float64)

    def train(self, data):
        yi = data[:self._rbf_ny] # 今のシステム出力
        ui = data[-self._rbf_nu:] # 今のシステム入力

        if len(self._past_sys_input) == self._past_sys_input_limit \
        and len(self._past_sys_output) == self._past_sys_output_limit:
            input = self._gen_input()
            start = time.time()
            now_y = self.update_rbf(input, yi)
            finish = time.time()
            self._update_rbf_time.append(finish - start)
            if self.realtime :
                self._train_pre_res.append(now_y)

        if len(self._past_sys_input) == self._past_sys_input_limit:
            del self._past_sys_input[:self._rbf_nu]
        self._past_sys_input.extend(ui)

        if len(self._past_sys_output) == self._past_sys_output_limit:
            del self._past_sys_output[:self._rbf_ny]
        self._past_sys_output.extend(yi)

    def test(self, data):
        yi = data[:self._rbf_ny]
        ui = data[-self._rbf_nu:]

        if len(self._past_sys_input) == self._past_sys_input_limit \
        and len(self._past_sys_output) == self._past_sys_output_limit:
            input = self._gen_input()
            self._test_pre_res.append(self._rbf.calc_f(input))

        if len(self._past_sys_input) == self._past_sys_input_limit:
            del self._past_sys_input[:self._rbf_nu]
        self._past_sys_input.extend(ui)

        if len(self._past_sys_output) == self._past_sys_output_limit:
            del self._past_sys_output[:self._rbf_ny]
        self._past_sys_output.extend(yi)

    def _save_hist(self, err_file, h_file):
        '''
        誤差履歴，隠れニューロン数履歴の保存
        '''
        with open(err_file, mode='w') as f:
            f.write(str(self._Nw)+'\n')
            f.write('\n'.join(map(str, self._Id_hist))+'\n')
        with open(h_file, mode='w') as f:
            f.write('\n'.join(map(str, self._h_hist))+'\n')

    def _save_pre_res(self, pre_res, ps_file):
        """
        予測結果の保存
        """
        with open(ps_file, mode='w') as f:
            for res in pre_res:
                f.write('\t'.join(map(str, res.tolist()))+'\n')

    def save_res(self, err_file, h_hist_file, test_ps_file, train_ps_file):
        self._save_hist(err_file, h_hist_file)
        if len(self._test_pre_res) :
            self._save_pre_res(self._test_pre_res, test_ps_file)
            print('Saved test prediction results as file: '+test_ps_file)
        if len(self._train_pre_res) :
            self._save_pre_res(self._train_pre_res, train_ps_file)
            print('Saved realtime prediction results as file: '+train_ps_file)

    def calc_MAE(self):
        """
        全履歴から評価指標のMAEを計算
        """
        return sum(self._ei_abs)/len(self._ei_abs)

    def calc_mean_update_time(self):
        """
        1回の更新にかかる時間の平均を計算
        """
        return sum(self._update_rbf_time)/len(self._update_rbf_time)

    def get_rbf(self):
        """
        rbfのネットワークのパラメータを返す
        パラメータを外部に渡すだけなので例外的にprivateメンバ変数に直接アクセスしている
        """
        self._rbf._gen_network_from_hidden_unit()
        dict = {
            'w0' : self._rbf._w0,
            'wk' : self._rbf._wk,
            'myu' : self._rbf._myu,
            'sigma' : self._rbf._sigma}

        return dict

if __name__ == "__main__" :
    np.set_printoptions(precision=6, suppress=True)
    rbf_mran = RBF_MRAN(
        nu=1, # システム入力(制御入力)の次元
        ny=1, # システム出力ベクトルの次元
        past_sys_input_num=1, # 過去のシステム入力保存数
        past_sys_output_num=1, # 過去のシステム出力保存数
        init_h=0, # スタート時の隠れニューロン数
        E1=0.01, E2=0.01, E3=1.2, Nw=48, Sw=48)

    start = time.time()
    rbf_mran.train('./data/siso/train.txt')
    duration = time.time()-start
    print('rbf_mran.train() duration[s]: ', str(duration))
    print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran._update_rbf_time)/len(rbf_mran._update_rbf_time))
    rbf_mran._save_hist('./model/history/siso/error.txt', './model/history/siso/h.txt')
    rbf_mran.test('./data/siso/test.txt')