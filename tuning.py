import os
import time
import optuna
import argparse

from RBF_MRAN import RBF_MRAN
from generate_data import gen_data
from utils import save_param, gen_study

class Objective(object):
    def __init__(self, study_folder):
        self.study_folder = study_folder
        self.train_file = study_folder+'/data/train.txt'
        self.test_file = study_folder+'/data/test.txt'
        self.train_ps_file = study_folder+'/data/train_pre_res.txt'
        self.test_ps_file = study_folder+'/data/test_pre_res.txt'
        self.err_file = study_folder+'/history/error.txt'
        self.h_hist_file = study_folder+'/history/h.txt'
        self.param_file = study_folder+'/model/param_by_optuna.yaml'
        print(self.train_file)
        self.min_MAE = 1e10

    def __call__(self, trial):
        psin = trial.suggest_int('past_sys_input_num', 1, 1)
        pson = trial.suggest_int('past_sys_output_num', 1, 5)
        E1 = trial.suggest_discrete_uniform('E1', 1e-3, 0.01, 1e-3)
        E2 = trial.suggest_discrete_uniform('E2', 1e-3, 0.01, 1e-3)
        E3_max = trial.suggest_discrete_uniform('E3_max', 0.5, 2.0, 0.1)
        E3_min = trial.suggest_discrete_uniform('E3_min', 0.1, E3_max, 0.1)
        gamma = trial.suggest_discrete_uniform('gamma', 0.95, 1.0, 0.01)
        kappa = trial.suggest_discrete_uniform('kappa', 0.1, 3, 0.01)
        # kappa = trial.suggest_int('kappa', 1, 1)
        # p0 = trial.suggest_discrete_uniform('p0', 0.1, 10, 0.1)
        p0 = trial.suggest_int('p0', 1, 1)
        q = trial.suggest_discrete_uniform('q', 0.01, 1, 0.01)
        Nw = trial.suggest_int('Nw', 10, 80)
        Sw = trial.suggest_int('Sw', 10, 80)

        with open(self.train_file, mode='r') as f:
            l = f.readlines()
        datas = [list(map(float, s.strip().split())) for s in l]

        rbf_mran = RBF_MRAN(
            nu=int(datas[2][0]), # システム入力(制御入力)の次元
            ny=int(datas[1][0]), # システム出力ベクトルの次元
            past_sys_input_num=psin, # 過去のシステム入力保存数
            past_sys_output_num=pson, # 過去のシステム出力保存数
            init_h=0, # スタート時の隠れニューロン数
            E1=E1,
            E2=E2,
            E3=-1,# 使わない
            E3_max=E3_max,
            E3_min=E3_min,
            kappa=kappa,
            p0=p0,
            q=q,
            gamma=gamma,
            Nw=Nw,
            Sw=Sw,
            study_folder=self.study_folder)

        print('Start train')
        start = time.time()
        # 学習
        for data in datas[int(datas[0][0])+1:] :
            rbf_mran.train(data)
            if time.time() - start > 3*3600 :
                print("Timeout...")
                return 1e5 # 時間がかかりすぎているので中止
        print('Finish train')

        MAE = rbf_mran.calc_MAE()
        if MAE < self.min_MAE:
            self.min_MAE = MAE
            save_param({
                    'E1': E1, 'E2': E2, 'E3': -1, 'E3_max': E3_max, 'E3_min': E3_min,
                    'Nw': Nw, 'Sw': Sw, 'gamma': gamma, 'kappa': kappa, 'init_h': 0,
                    'p0': p0, 'q': q, 'past_sys_input_num': psin, 'past_sys_output_num': pson
                },
                self.param_file)

            with open(self.test_file, mode='r') as f:
                l = f.readlines()
            datas = [s.strip().split() for s in l]

            print('Start test')
            # 検証
            for data in datas[int(datas[0][0])+1:] :
                rbf_mran.test(data)
            print('Finish test')

            rbf_mran.save_res(is_last_save=True)

        return MAE

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)')
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-nt', '--n_trials', type=int, required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/test data.', action='store_true')
    parser.add_argument('-dl', '--data_len', type=int, default=5000)
    # parser.add_argument('-rt', '--realtime', help='Trueなら，学習中（＝リアルタイムのシステム同定）の履歴を保存する．Default: False.', action='store_true')
    args = parser.parse_args()

    # プロジェクトフォルダ作成
    study_folder = gen_study(args.study_name)

    # データ生成
    train_file = study_folder+'/data/train.txt'
    test_file = study_folder+'/data/test.txt'
    if args.gen_new_data or not os.path.isfile(train_file) :
        gen_res = gen_data(args.sys, train_file, test_file, args.data_len)
        if gen_res < 0 :
            exit()

    study = optuna.create_study(
        study_name=args.study_name,
        storage='sqlite:///'+study_folder+'/model/param.db',
        load_if_exists=True)
    study.optimize(Objective(study_folder), n_trials=args.n_trials)

    # 最適パラメータの保存
    param = {}
    for key, value in study.best_trial.params.items():
        param[key] = value
    param['init_h'] = 0 # プログラムの都合上追記しとく
    param['E3'] = -1 # プログラムの都合上追記しとく

    save_param(param, study_folder+'/model/best_param_by_optuna.yaml')
