'''
学習用データで学習し，検証データで検証するプログラム
'''

import argparse
import os
import tqdm
import numpy as np

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_study
from utils import load_param, gen_study, save_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', required=True)
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/test data. Default: False.', action='store_true')
    parser.add_argument('-dl', '--data_len', help='生成されるデータファイルの行数．', type=int, default=5000)
    parser.add_argument('-ncx', '--n_change_x', help='xが切り替わるnの指定．', type=int, default=-1)
    parser.add_argument('-ncu', '--n_change_u', help='uが切り替わるnの指定．', type=int, default=-1)
    parser.add_argument('-pf', '--param_file', help='モデル初期化の際に用いるハイパーパラメータファイル名．', default='param.yaml')
    parser.add_argument('-rt', '--realtime', help='Trueなら，学習中（＝リアルタイムのシステム同定）の履歴を保存する．Default: False.', action='store_true')
    parser.add_argument('-ps', '--plot_start', help='See plot.py/plot_pre_res() doc string.', type=int, default=3500)
    parser.add_argument('-pl', '--plot_len', help='See plot.py/plot_pre_res() doc string.', type=int, default=500)
    parser.add_argument('-pm', '--plot_mode', help='See plot.py/plot_err_hist() code.', type=int, default=1)
    parser.add_argument('-id', '--input_delay', type=int, default=0)
    parser.add_argument('-od', '--output_delay', type=int, default=0)
    parser.add_argument('-xb', '--x_before', type=int, default=-1)
    parser.add_argument('-xa', '--x_after', type=int, default=-1)
    parser.add_argument('-ub', '--u_before', type=int, default=-1)
    parser.add_argument('-ua', '--u_after', type=int, default=-1)
    args = parser.parse_args()

    # プロジェクトフォルダ作成
    study_folder = gen_study(args.study_name)

    # 引数の保存
    save_args(args, study_folder+'/latest_example_args.yaml')

    # データ生成
    train_file = study_folder+'/data/train.txt'
    test_file = study_folder+'/data/test.txt'
    if args.gen_new_data or not os.path.isfile(train_file) :
        gen_res = gen_data(
            sys_type=args.sys, train_file=train_file, test_file=test_file,
            data_len=args.data_len, ncx=args.n_change_x, ncu=args.n_change_u,
            xb=args.x_before, xa=args.x_after, ub=args.u_before, ua=args.u_after)
        if gen_res < 0 :
            exit()

    # パラメータファイル生成or読み込み
    param_file = study_folder+'/model/'+args.param_file
    if args.param_file == 'param.yaml' and not os.path.isfile(param_file) :
        gen_param_file_from_cmd(param_file)

    param = load_param(param_file)
    with open(train_file, mode='r') as f:
        l = f.readlines()
    datas = [list(map(float, s.strip().split())) for s in l]

    rbf_mran = RBF_MRAN(
        nu=int(datas[2][0]), # システム入力(制御入力)の次元
        ny=int(datas[1][0]), # システム出力ベクトルの次元
        past_sys_input_num=param['past_sys_input_num'], # 過去のシステム入力保存数
        past_sys_output_num=param['past_sys_output_num'], # 過去のシステム出力保存数
        init_h=param['init_h'], # スタート時の隠れニューロン数
        E1=param['E1'],
        E2=param['E2'],
        E3=param['E3'],
        E3_max=param['E3_max'],
        E3_min=param['E3_min'],
        gamma=param['gamma'],
        Nw=param['Nw'],
        Sw=param['Sw'],
        kappa=param['kappa'] if 'kappa' in param else 1.0,
        realtime=args.realtime,
        input_delay=args.input_delay, # 入力の遅れステップ
        output_delay=args.output_delay) # 出力の観測の遅れステップ

    # 学習
    print('Start train.')
    for data in tqdm.tqdm(datas[int(datas[0][0])+1:]) :
        rbf_mran.train(data)
    print('Train finished.')
    print('mean rbf_mran.update_rbf() duration[s]: ', rbf_mran.calc_mean_update_time())
    print('Total MAE: ', rbf_mran.calc_MAE())

    if not args.realtime :
        with open(test_file, mode='r') as f:
            l = f.readlines()
        datas = [s.strip().split() for s in l]
        # 検証
        print('Start test.')
        for data in tqdm.tqdm(datas[int(datas[0][0])+1:]) :
            rbf_mran.test(data)
        print('Test finished.')

    os.remove('tmp.txt')
    def savetxt(fh, array):
        np.savetxt(fh, array, fmt='% .18e', delimiter = "\t")
        fh.write("\n")
    param = rbf_mran.get_rbf()
    with open('tmp.txt', 'a') as f :
        savetxt(f, param['w0'])
        savetxt(f, param['wk'])
        savetxt(f, param['myu'])
        savetxt(f, param['sigma'])

    # 色々保存とプロット
    rbf_mran.save_res(
        err_file=study_folder+'/history/error.txt',
        h_hist_file=study_folder+'/history/h.txt',
        test_ps_file=study_folder+'/data/test_pre_res.txt',
        train_ps_file=study_folder+'/data/train_pre_res.txt')
    plot_study(
        study_name=args.study_name,
        plot_start=args.plot_start,
        plot_len=args.plot_len,
        need_rt=args.realtime,
        eh_mode=args.plot_mode)




