'''
リアルタイムでシステムが変わっても同定できるかどうかを試すプログラム
'''

import argparse
import os

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_all
from utils import load_param, gen_study

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data. Default: False.', action='store_true')
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-dl', '--data_len', help='生成されるデータファイルの行数', type=int, default=5000)
    parser.add_argument('-ncx', '--n_change_x', help='xの生成式を切り替える境目', type=int, default=-1)
    parser.add_argument('-ncu', '--n_change_u', help='uの生成式を切り替える境目', type=int, default=-1)
    parser.add_argument('-ps', '--plot_start', type=int, default=3500)
    parser.add_argument('-pl', '--plot_len', type=int, default=500)
    parser.add_argument('-pf', '--param_file', default='param.yaml')
    args = parser.parse_args()

    # プロジェクトフォルダ作成
    study_folder = './study/'+args.study_name
    gen_study(study_folder)

    # データ生成
    train_file = study_folder+'/data/train.txt'
    val_file = study_folder+'/data/val.txt' # 使われないがとりあえず生成
    if args.gen_new_data or not os.path.isfile(train_file) :
        gen_res = gen_data(
            args.sys, train_file, val_file,
            args.data_len, ncx=args.n_change_x, ncu=args.n_change_u)
        if gen_res < 0 :
            exit()

    with open(train_file, mode='r') as f:
        l = f.readlines()
    datas = [list(map(float, s.strip().split())) for s in l]

    # パラメータファイル生成or読み込み
    param_file = study_folder+'/model/'+args.param_file
    if args.param_file == 'param.yaml' and not os.path.isfile(param_file) :
        gen_param_file_from_cmd(param_file)

    param = load_param(param_file)
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
        Sw=param['Sw'])
    
    # 学習（リアルタイムでのシステム同定）
    print('Start real time train.')
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.train(data, True)
    print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran.update_rbf_time)/len(rbf_mran.update_rbf_time))
    print('Total MAE: ', rbf_mran.calc_MAE())
    
    # 色々保存とプロット
    pre_res_file = study_folder+'/data/pre_res.txt'
    err_file = study_folder+'/history/error.txt'
    h_hist_file = study_folder+'/history/h.txt'
    rbf_mran.save_res(err_file, h_hist_file, pre_res_file)
    plot_all(
        err_file=err_file,
        h_hist_file=h_hist_file,
        val_file=train_file, # リアルタイムのシステム同定結果を見たいので，これはタイプミスではない
        pre_res_file=pre_res_file,
        plot_start=args.plot_start,
        plot_len=args.plot_len,
        mode=1)


