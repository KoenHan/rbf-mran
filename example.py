'''
学習用データで学習し，検証データで検証するプログラム
'''

import argparse
import os

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_study
from utils import load_param, gen_study

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', required=True)
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data. Default: False.', action='store_true')
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-dl', '--data_len', help='生成されるデータファイルの行数', type=int, default=5000)
    parser.add_argument('-pf', '--param_file', default='param.yaml')
    parser.add_argument('-rt', '--realtime', help='Trueなら，学習中（＝リアルタイムのシステム同定）の履歴を保存する．Default: False', action='store_true')
    args = parser.parse_args()

    # プロジェクトフォルダ作成
    study_folder = gen_study(args.study_name)

    # データ生成
    train_file = study_folder+'/data/train.txt'
    val_file = study_folder+'/data/val.txt'
    if args.gen_new_data or not os.path.isfile(train_file) :
        gen_res = gen_data(args.sys, train_file, val_file, args.data_len)
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
        Sw=param['Sw'],
        realtime=args.realtime)
    
    # 学習
    print('Start train.')
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.train(data)
    print('Train finished.')
    print('mean rbf_mran.update_rbf() duration[s]: ', rbf_mran.calc_mean_update_time())
    print('Total MAE: ', rbf_mran.calc_MAE())
    
    with open(val_file, mode='r') as f:
        l = f.readlines()
    datas = [s.strip().split() for s in l]

    # 検証
    print('Start validation.')
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.val(data)
    print('Validation finished.')

    # 色々保存とプロット
    rbf_mran.save_res(
        err_file=study_folder+'/history/error.txt',
        h_hist_file=study_folder+'/history/h.txt',
        val_ps_file=study_folder+'/data/val_pre_res.txt',
        rt_ps_file=study_folder+'/data/rt_pre_res.txt')
    plot_study(
        study_name=args.study_name, 
        plot_start=3500,
        plot_len=500,
        need_rt=args.realtime,
        eh_mode=1)



