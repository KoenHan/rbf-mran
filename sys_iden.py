'''
学習用データで学習し，検証データで検証するプログラム
'''

import argparse
import os
import tqdm

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_study
from utils import load_param, gen_study, save_args

def main(study_folder='./ros_test'):
    param_file = study_folder+'/model/param.yaml'
    param = load_param(param_file)
    rbf_mran = RBF_MRAN(
        nu=4, # システム入力(制御入力)の次元=各ロータへの入力
        ny=10, # システム出力ベクトルの次元 todo:不明なので調べる
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

if __name__ == '__main__':
    main()