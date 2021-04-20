import optuna
import argparse

from RBF_MRAN import RBF_MRAN
from generate_data import generate_data

# あとで書き換えられる
TRAIN_FILE = './data/mimo/optuna/train.txt'
VAL_FILE = './data/mimo/optuna/val.txt' # valは今の所使わない

def objective(trial):
    # psin = trial.suggest_int('past_sys_input_num', 1, 3)
    # pson = trial.suggest_int('past_sys_output_num', 1, 3)
    psin = 1
    pson = 2
    E1 = trial.suggest_uniform('E1', 0, 1)
    E2 = trial.suggest_uniform('E2', 0, 1)
    E3_max = trial.suggest_uniform('E3_max', 0, 3)
    E3_min = trial.suggest_uniform('E3_min', 0, 3)
    gamma = trial.suggest_discrete_uniform('gamma', 0.990, 1.001, 0.001)
    Nw = trial.suggest_int('Nw', 1, 100)
    Sw = trial.suggest_int('Sw', 1, 100)

    with open(TRAIN_FILE, mode='r') as f:
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
    # print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran.update_rbf_time)/len(rbf_mran.update_rbf_time))
    
    with open(VAL_FILE, mode='r') as f:
        l = f.readlines()
    datas = [s.strip().split() for s in l]

    return rbf_mran.calc_MAE()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', required=True)
    parser.add_argument('-tf', '--train_file', required=True)
    parser.add_argument('-vf', '--val_file', required=True)
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-tt', '--train_time', type=int, required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data.', action='store_true')
    args = parser.parse_args()

    if args.gen_new_data :
        gen_res = generate_data(args.sys, TRAIN_FILE, VAL_FILE, args.train_time)
        if gen_res < 0 :
            exit()

    TRAIN_FILE = args.train_file
    VAL_FILE = args.val_file
    study_name = args.study_name
    db_file = './model/param/'+study_name+'.db'
    study = optuna.create_study(
        study_name=study_name,
        storage='sqlite:///'+db_file,
        load_if_exists=True)
    study.optimize(objective, n_trials=1000)
