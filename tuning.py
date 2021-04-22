import os
import yaml
import optuna
import argparse

from RBF_MRAN import RBF_MRAN
from generate_data import gen_data
from utils import save_param

class Objective(object):
    def __init__(self, project_folder):
        self.train_file = project_folder+'/data/train.txt'
        self.val_file = project_folder+'/data/val.txt'
        self.pre_res_file = project_folder+'/data/pre_res.txt'
        self.err_file = project_folder+'/history/error.txt'
        self.h_hist_file = project_folder+'/history/h.txt'
        self.param_file = project_folder+'/model/param.yaml'
        print(self.train_file)
        self.min_MAE = 1e10
    
    def __call__(self, trial):
        psin = trial.suggest_int('past_sys_input_num', 1, 1)
        pson = trial.suggest_int('past_sys_output_num', 2, 2)
        E1 = trial.suggest_discrete_uniform('E1', 0, 1.0, 1e-4)
        E2 = trial.suggest_discrete_uniform('E2', 0, 1.0, 1e-4)
        E3_max = trial.suggest_discrete_uniform('E3_max', 0, 3.0, 1e-4)
        E3_min = trial.suggest_discrete_uniform('E3_min', 0, E3_max, 1e-4)
        gamma = trial.suggest_discrete_uniform('gamma', 0.990, 1.0, 0.001)
        Nw = trial.suggest_int('Nw', 1, 100)
        Sw = trial.suggest_int('Sw', 1, 100)

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
            gamma=gamma,
            Nw=Nw,
            Sw=Sw)
        
        # 学習
        for data in datas[int(datas[0][0])+1:] :
            rbf_mran.train(data)
        
        with open(self.val_file, mode='r') as f:
            l = f.readlines()
        datas = [s.strip().split() for s in l]

        MAE = rbf_mran.calc_MAE()
        if MAE < self.min_MAE:
            self.min_MAE = MAE
            with open(self.val_file, mode='r') as f:
                l = f.readlines()
            datas = [s.strip().split() for s in l]

            # 検証
            for data in datas[int(datas[0][0])+1:] :
                rbf_mran.val(data)
                
            rbf_mran.save_res(self.err_file, self.h_hist_file, self.pre_res_file)
            save_param({
                    'E1': E1, 'E2': E2, 'E3': -1, 'E3_max': E3_max, 'E3_min': E3_min,
                    'Nw': Nw, 'Sw': Sw, 'gamma': gamma, 'init_h': 0,
                    'past_sys_input_num': psin, 'past_sys_output_num': pson
                },
                self.param_file)

        return MAE

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', required=True)
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-nt', '--n_trials', type=int, required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data.', action='store_true')
    parser.add_argument('-dl', '--data_len', type=int, default=5000)
    args = parser.parse_args()

    # プロジェクトフォルダ作成
    project_folder = './study/'+args.study_name
    for directory in ['/data', '/model', '/history']:
        fpath = project_folder+directory
        if not os.path.isdir(fpath):
            os.makedirs(fpath)

    # データ生成
    train_file = project_folder+'/data/train.txt'
    val_file = project_folder+'/data/val.txt'
    if args.gen_new_data or not os.path.isfile(train_file) :
        gen_res = gen_data(args.sys, train_file, val_file, args.data_len)
        if gen_res < 0 :
            exit()

    study = optuna.create_study(
        study_name=args.study_name,
        storage='sqlite:///'+project_folder+'/model/param.db',
        load_if_exists=True)
    study.optimize(Objective(project_folder), n_trials=args.n_trials)

    # 最適パラメータの保存
    param = {}
    for key, value in study.best_trial.params.items():
        param[key] = value
    param['init_h'] = 0 # プログラムの都合上追記しとく
    param['E3'] = -1 # プログラムの都合上追記しとく

    save_param(param, project_folder+'/model/param.yaml')

