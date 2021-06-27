import enum
import numpy as np
import time
import os

from RBF import RBF
from rosgraph.network import is_local_address

class RBF_MRAN:
    def __init__(self, nu, ny, past_sys_input_num, past_sys_output_num,
            init_h, E1, E2, E3, E3_max, E3_min, gamma, Nw, Sw, kappa=1.0,
            p0=1, q=0.1, realtime=False, input_delay=0, output_delay=0,
            study_folder=None):
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

        # self._ei_abs = [] # 学習中の誤差履歴(MAE)
        self._ei_abs = [0 for _ in range(self._Nw)] # 全部記録するのはメモリを使いすぎるのでこっちにする
        self._ei_abs_idx = 0
        self._ei_abs_sum = 0.0
        self._Id_hist = [] # 学習中の誤差履歴(式3.16)
        self._h_hist = [init_h] # 学習中の隠れニューロン数履歴
        self._train_pre_res = [] # リアルタイムのシステム同定の結果の保存
        self._test_pre_res = [] # 検証時の予測結果の保存

        self._total_MAE = 0.0 # 学習全体のMAE計算用
        self._cnt_train_num = 0 # 学習回数のカウント（学習全体のMAE計算時に用いる）
        # self._update_rbf_time = [] # 時間計測
        self._update_rbf_time_sum = 0.0 # 時間計測
        self.realtime = realtime # リアルタイムのシステム同定の場合のフラグ

        # 各種データ保存先
        self.err_file = study_folder+'/history/error.txt'
        self.h_hist_file = study_folder+'/history/h.txt'
        self.test_ps_file = study_folder+'/data/test_pre_res.txt'
        self.train_ps_file = study_folder+'/data/train_pre_res.txt'

        # 既存のデータファイルの削除
        for i, file in enumerate([self.err_file, self.h_hist_file, self.test_ps_file, self.train_ps_file]) :
            if os.path.isfile(file) :
                os.remove(file)
            if i == 0 :
                with open(file, 'w') as f:
                    f.write(str(self._Nw)+'\n')

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

    @profile
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

        # 更新が終わったのでインクリメント
        self._cnt_train_num += 1

        # 学習中の誤差（式3.16）の算出及び保存
        # self._ei_abs.append(ei_norm)
        # if len(self._ei_abs) >= self._Nw:
        #     self._Id_hist.append(sum(self._ei_abs[-self._Nw:])/self._Nw)
        idx = self._ei_abs_idx%self._Nw
        self._ei_abs_sum -= self._ei_abs[idx]
        self._ei_abs_sum += ei_norm
        self._ei_abs[idx] = ei_norm
        self._ei_abs_idx += 1
        if self._ei_abs_idx >= self._Nw :
            self._Id_hist.append(self._ei_abs_sum/self._Nw)
        # 学習中の隠れニューロン数保存
        self._h_hist.append(self._rbf.get_h())
        # 学習全体のMAEの計算のために合計を取る
        self._total_MAE += ei_norm

        # 履歴の逐次保存
        self.save_res()

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
            # self._update_rbf_time.append(finish - start)
            self._update_rbf_time_sum += finish - start
            if self.realtime :
                self._train_pre_res.append(now_y)
                if len(self._train_pre_res) >= 500 :
                    with open(self.train_data_file, 'a') as f:
                        for d in self.data:
                            f.write('\t'.join(list(map(str, d)))+'\n')

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

    def _save_hist(self, is_last_save=False):
        '''
        誤差履歴，隠れニューロン数履歴の逐次保存
        '''
        if len(self._Id_hist) >= 500 or is_last_save :
            with open(self.err_file, mode='a') as f:
                f.write('\n'.join(map(str, self._Id_hist))+'\n')
            self._Id_hist = []
        if len(self._h_hist) >= 500 or is_last_save :
            with open(self.h_hist_file, mode='a') as f:
                f.write('\n'.join(map(str, self._h_hist))+'\n')
            self._h_hist = []

    def _save_pre_res(self, pre_res, ps_file, is_last_save=False):
        """
        予測結果の逐次保存
        """
        if len(pre_res) >= 500 or is_last_save :
            with open(ps_file, mode='a') as f:
                for res in pre_res:
                    f.write('\t'.join(map(str, res.tolist()))+'\n')
            pre_res = []

    def save_res(self, is_last_save=False):
        if len(self._h_hist) :
            self._save_hist(is_last_save)
        if len(self._test_pre_res) :
            self._save_pre_res(self._test_pre_res, self.test_ps_file, is_last_save)
        if len(self._train_pre_res) :
            self._save_pre_res(self._train_pre_res, self.train_ps_file, is_last_save)

    def calc_MAE(self):
        """
        全履歴から評価指標のMAEを計算
        """
        # return sum(self._ei_abs)/len(self._ei_abs)
        return self._total_MAE/self._cnt_train_num

    def calc_mean_update_time(self):
        """
        1回の更新にかかる時間の平均を計算
        """
        # return sum(self._update_rbf_time)/len(self._update_rbf_time)
        return self._update_rbf_time_sum/self._cnt_train_num

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