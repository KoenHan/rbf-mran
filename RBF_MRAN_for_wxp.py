import numpy as np
import time
import os

from RBF import RBF
from utils import save_ndarray, load_ndarray

class RBF_MRAN:
    def __init__(self, nu, ny, past_sys_input_num, past_sys_output_num,
            init_h, E1, E2, E3, E3_max, E3_min, gamma, Nw, Sw, kappa=1.0,
            p0=1, q=0.1, input_delay=0, output_delay=0, study_folder=None,
            use_exist_net=False, readonly=False) :
        # 各種データ保存先
        self._err_file = study_folder+'/history/error.txt'
        self._h_hist_file = study_folder+'/history/h.txt'
        self._test_ps_file = study_folder+'/data/test_pre_res.txt'
        self._train_ps_file = study_folder+'/data/train_pre_res.txt'
        self._w0_param_file = study_folder+'/model/w0.txt'
        self._wk_param_file = study_folder+'/model/wk.txt'
        self._myu_param_file = study_folder+'/model/myu.txt'
        self._sigma_param_file = study_folder+'/model/sigma.txt'

        init_w0 = None
        init_wk = None
        init_myu = None
        init_sigma = None
        if use_exist_net :
            init_w0 = load_ndarray(self._w0_param_file)
            init_wk = load_ndarray(self._wk_param_file)
            if len(init_wk.shape) == 1 :
                init_wk = init_wk.reshape(1, -1)
            init_myu = load_ndarray(self._myu_param_file)
            if len(init_myu.shape) == 1 :
                init_myu = init_myu.reshape(1, -1)
            init_sigma = load_ndarray(self._sigma_param_file)
            init_h = init_wk.shape[1]
        # RBFネットワーク初期化
        self._rbf = RBF(
            nu=nu, ny=ny, init_h=init_h,
            input_size = past_sys_input_num*nu + past_sys_output_num*ny,
            use_exist_net = use_exist_net,
            init_w0=init_w0,
            init_wk=init_wk,
            init_myu=init_myu,
            init_sigma=init_sigma)

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
        self._pre_yi = np.zeros(self._rbf_ny, dtype=np.float64)

        # Step 2で使われるパラメータ
        self._kappa = kappa

        # Step 4で使われるパラメータ
        #  全部とりあえずの値
        self._p0 = p0
        self._P = np.eye(self._rbf.get_param_num())
        self._q = q # 小さくする
        # self._q = 0.1
        # self._R = np.eye(self._rbf_ny, dtype=np.float64) # 観測誤差ノイズ 大きくする
        self._R = np.zeros((self._rbf_ny, self._rbf_ny), dtype=np.float64) # 観測誤差ノイズ 大きくする

        # Step 5で使われるパラメータ
        self._delta = 1e-9 # p.55
        # self._delta = 0.0001 # p.55
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

        self.readonly = readonly # 消してほしくない時に

        if not self.readonly :
            # 既存のデータファイルの削除
            for i, file in enumerate([self._err_file, self._h_hist_file, self._test_ps_file, self._train_ps_file]) :
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

    def _calc_error_criteria(self, input, f, yi_np):
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
        ei = yi_np - f
        ei_norm = np.linalg.norm(ei, ord=2)
        # 学習の改善のために以下のエラーを追加する
        # ei_norm += np.linalg.norm(self._pre_yi - yi_np, ord=2)/(np.linalg.norm(self._pre_yi - f, ord=2) + 1)
        di = self._rbf.get_closest_unit_myu_and_dist(input)

        self._past_ei_norm_pow.append(ei@ei)
        # self._past_ei_norm_pow.append(ei_norm**2)
        if len(self._past_ei_norm_pow) > self._Nw :
            del self._past_ei_norm_pow[0]

        case = 0
        if ei_norm <= self._E1 :
            case = 1
        elif sum(self._past_ei_norm_pow) <= self._E2_pow*self._Nw :
            case = 2
        elif di <= self._calc_E3(): # p.55の実験に合わせた
            case = 3

        '''
        memo: 返すeiについて
        エラーに第2項を追加したが，
        追加した項をどうeiに反映させればいいのかわからないので現状維持
        もうちょっと考えたほうが良さそう
        '''
        return case, ei, ei_norm, di

    def update_rbf(self, input, yi_np):
        f = self._rbf.calc_f(input)
        o = self._rbf.calc_o()

        # Step 1
        satisfied, ei, ei_norm, di = self._calc_error_criteria(input, f, yi_np)
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

        return f

    def _gen_input(self):
        return np.array(self._past_sys_output[:self._past_sys_output_size]
            + self._past_sys_input[:self._past_sys_input_size], dtype=np.float64)
        # return np.array([0 for _ in range(self._past_sys_output_size)]
        #     + self._past_sys_input[:self._past_sys_input_size], dtype=np.float64) # wxpの学習ではpsoは必要ないので

    def train(self, data):
        yi = data[:self._rbf_ny] # 今のシステム出力
        yi_np = np.array(yi, dtype=np.float64)
        ui = data[-self._rbf_nu:] # 今のシステム入力



        if len(self._past_sys_input) == self._past_sys_input_limit \
        and len(self._past_sys_output) == self._past_sys_output_limit:
            input = self._gen_input()
            start = time.time()
            now_y = self.update_rbf(input, yi_np)
            finish = time.time()
            self._update_rbf_time_sum += finish - start
            self._train_pre_res.append(now_y)

        '''
        以下の3行を236行目あたりに移動してもう一回予測を行う
        式3.3のuとyが左下のような関係ならこのままでいいが，今回は右下でなのでこのままでは良くない
              y   u           y   u
        t   : ○→○          ○←○
                ↙            (tとt+1の間では制御的には無関係）
        t+1 : ○→○          ○←○
        '''

        if len(self._past_sys_output) == self._past_sys_output_limit:
            del self._past_sys_output[:self._rbf_ny]
        self._past_sys_output.extend(yi)

        if len(self._past_sys_input) == self._past_sys_input_limit:
            del self._past_sys_input[:self._rbf_nu]
        self._past_sys_input.extend(ui)

        self.save_res() # 履歴の逐次保存

        self._pre_yi = yi_np

    def test(self, data):
        yi = data[:self._rbf_ny]
        ui = data[-self._rbf_nu:]



        if len(self._past_sys_input) == self._past_sys_input_limit \
        and len(self._past_sys_output) == self._past_sys_output_limit:
            input = self._gen_input()
            # print('input ', input)
            self._test_pre_res.append(self._rbf.calc_f(input))
            # print('tpr', self._test_pre_res[-1].tolist())
            # print(len(self._test_pre_res))
        '''
        以下の3行を261行目あたりに移動してもう一回予測を行う
        式3.3のuとyが左下のような関係ならこのままでいいが，今回は右下でなのでこのままでは良くない
              y   u           y   u
        t   : ○→○          ○←○
                ↙            (tとt+1の間では制御的には無関係）
        t+1 : ○→○          ○←○
        '''

        if len(self._past_sys_output) == self._past_sys_output_limit:
            del self._past_sys_output[:self._rbf_ny]
        self._past_sys_output.extend(yi)

        if len(self._past_sys_input) == self._past_sys_input_limit:
            del self._past_sys_input[:self._rbf_nu]
        self._past_sys_input.extend(ui)
        # print('psi ', self._past_sys_input)
        # print('pso ', self._past_sys_output)

        self.save_res() # 履歴の逐次保存

    def save_res(self, is_last_save=False):
        # len(*) == 0ならis_last_saveを無視したいからこうしてるけど書き直せるけど一旦放置
        if len(self._h_hist) :
            self._save_hist(is_last_save)
        if len(self._test_pre_res) :
            self._save_pre_res(self._test_pre_res, self._test_ps_file, is_last_save, 'test')
        if len(self._train_pre_res) :
            self._save_pre_res(self._train_pre_res, self._train_ps_file, is_last_save, 'train')
        if is_last_save and not self.readonly:
            self._save_param()

    def _save_hist(self, is_last_save=False):
        '''
        誤差履歴，隠れニューロン数履歴の逐次保存
        '''
        if len(self._Id_hist) >= 500 or is_last_save :
            if not self.readonly :
                with open(self._err_file, mode='a') as f:
                    f.write('\n'.join(map(str, self._Id_hist))+'\n')
            self._Id_hist = []
        if len(self._h_hist) >= 500 or is_last_save :
            if not self.readonly :
                with open(self._h_hist_file, mode='a') as f:
                    f.write('\n'.join(map(str, self._h_hist))+'\n')
            self._h_hist = []

    def _save_pre_res(self, pre_res, ps_file, is_last_save=False, type='test'):
        """
        予測結果の逐次保存
        """
        if len(pre_res) >= 500 or is_last_save :
            if not self.readonly :
                with open(ps_file, mode='a') as f:
                    for res in pre_res:
                        f.write('\t'.join(map(str, res.tolist()))+'\n')
            if type == 'test' :
                self._test_pre_res = []
            elif type == 'train' :
                self._train_pre_res = []


    def _save_param(self):
        """
        パラメータの保存
        パラメータを保存するだけなので例外的にprivateメンバ変数に直接アクセスしている
        """
        def sp(fn, param) :
            if os.path.isfile(fn) :
                os.remove(fn)
            with open(fn, 'w') as f :
                save_ndarray(f, param)

        self._rbf._gen_network_from_hidden_unit()
        sp(self._w0_param_file, self._rbf._w0)
        sp(self._wk_param_file, self._rbf._wk)
        sp(self._myu_param_file, self._rbf._myu)
        sp(self._sigma_param_file, self._rbf._sigma)

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

    def get_rbf_param(self):
        """
        rbfネットワークのパラメータを返す
        パラメータを外部に渡すだけなので例外的にprivateメンバ変数に直接アクセスしている
        """
        self._rbf._gen_network_from_hidden_unit()
        dict = {
            'h' : self._rbf._h,
            'w0' : self._rbf._w0,
            'wk' : self._rbf._wk,
            'myu' : self._rbf._myu,
            'sigma' : self._rbf._sigma}

        return dict

    def get_rbf_config(self):
        """
        rbfネットワークの設定を返す
        パラメータを外部に渡すだけなので例外的にprivateメンバ変数に直接アクセスしている
        """
        dict = {
            'ny' : self._rbf._ny,
            'nu' : self._rbf._nu,
            'psin' : self._past_sys_input_num,
            'pson' : self._past_sys_output_num}

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