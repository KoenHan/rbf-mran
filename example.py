import numpy as np
import argparse

from RBF_MRAN import RBF_MRAN
from generate_data import generate_data
from plot import plot_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', default='siso')
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data. Default: False.', action='store_true')
    parser.add_argument('-tf', '--train_file', default='train.txt')
    parser.add_argument('-vf', '--val_file', default='val.txt')
    args = parser.parse_args()

    train_file = './data/'+args.sys+'/'+args.train_file
    val_file = './data/'+args.sys+'/'+args.val_file
    if args.gen_new_data :
        gen_res = generate_data(args.sys, train_file, val_file)
        if gen_res < 0 :
            exit()

    # 学習
    with open(train_file, mode='r') as f:
        l = f.readlines()
    datas = [list(map(float, s.strip().split())) for s in l]

    rbf_mran = RBF_MRAN(
        nu=int(datas[2][0]), # システム入力(制御入力)の次元
        ny=int(datas[1][0]), # システム出力ベクトルの次元
        past_sys_input_num=1, # 過去のシステム入力保存数
        past_sys_output_num=1, # 過去のシステム出力保存数
        init_h=0, # スタート時の隠れニューロン数
        E1=0.01, E2=0.01, E3=1.2, Nw=48, Sw=48)
    
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.train(data)
    print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran.update_rbf_time)/len(rbf_mran.update_rbf_time))
    
    # 検証
    with open(val_file, mode='r') as f:
        l = f.readlines()
    datas = [s.strip().split() for s in l]

    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.val(data)
    
    # 色々保存とプロット
    pre_res_file = './data/'+args.sys+'/pre_res.txt'
    err_file = './model/history/'+args.sys+'/error.txt'
    h_hist_file = './model/history/'+args.sys+'/h.txt'
    rbf_mran.save_res(err_file, h_hist_file, pre_res_file)
    plot_all(
        err_file=err_file,
        h_hist_file=h_hist_file,
        val_file=val_file,
        pre_res_file=pre_res_file,
        mode=1)


