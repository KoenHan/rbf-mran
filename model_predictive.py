import argparse
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_study
from utils import load_param, gen_study, save_args, save_ndarray

LINEWIDTH = 0.8
WINWIDTH = 15
WINHEIGHT = 15

from scipy.spatial.transform import Rotation as R

def quaternion_to_euler(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]]) # x, y, z, w
    return r.as_euler('zyx', degrees=True) # z:yaw?, y:pitch?, x:roll? degrees=Trueでラジアン

def plot_res(x, y, label):
    # plt.xticks([i for i in range(0, len_data + 1, 300)])
    # plt.yticks([i for i in range(0, max(data)+1, 1)])
    plt.plot(x, y, label=label, linewidth=LINEWIDTH)
    plt.grid()
    plt.legend()
    # plt.title(title, y=-0.25)
    # plt.savefig(fig_folder+'/h_hist.png')

if __name__=="__main__" :
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-sn', '--study_name', required=True)
    # args = parser.parse_args()

    study_folder = "./study/ros_test"

    param_file = study_folder+'/model/param.yaml'

    train_file = study_folder+'/data/train.txt'
    with open(train_file, mode='r') as f:
        l = f.readlines()
    train_datas = [list(map(float, s.strip().split())) for s in l]

    qrs_file = study_folder+'/data/quat_rate_sysin.txt'
    with open(qrs_file, mode='r') as f:
        l = f.readlines()
    qrs_datas = [list(map(float, s.strip().split())) for s in l]

    param = load_param(param_file)
    hist_len = param['past_sys_output_num']
    rbf_mran = RBF_MRAN(
        nu=int(train_datas[2][0]), # システム入力(制御入力)の次元
        ny=int(train_datas[1][0]), # システム出力ベクトルの次元
        past_sys_input_num=param['past_sys_input_num'], # 過去のシステム入力保存数
        past_sys_output_num=hist_len, # 過去のシステム出力保存数
        # init_h=param['init_h'], # スタート時の隠れニューロン数
        init_h=0, # スタート時の隠れニューロン数
        E1=param['E1'],
        E2=param['E2'],
        E3=param['E3'],
        E3_max=param['E3_max'],
        E3_min=param['E3_min'],
        gamma=param['gamma'],
        Nw=param['Nw'],
        Sw=param['Sw'],
        kappa=param['kappa'] if 'kappa' in param else 1.0,
        p0=param['p'] if 'p' in param else 1.0,
        q=param['q'] if 'q' in param else 0.1,
        study_folder=study_folder,
        use_exist_net=True) # 既存のネットワークを使うかどうか

    start = 65000
    idx = int(qrs_datas[0][0]) + start
    horizen = 10
    y = qrs_datas[idx-hist_len:idx+horizen]
    for data in y[:-1] : # 加速度を学習したので長さはhorizen-1でいい
        rbf_mran.test(data[4:])

    title = "モデル予測結果"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))

    x = [i for i in range(start, start+horizen)]
    y1_roll = []
    y1_pitch = []
    y1_yaw = []
    for data in y[hist_len:hist_len+horizen] :
        eular = quaternion_to_euler(data[:4])
        y1_roll.append(eular[2])
        y1_pitch.append(eular[1])
        y1_yaw.append(eular[0])
    plot_res(x, y1_roll, "真値 roll")
    plot_res(x, y1_pitch, "真値 pitch")
    plot_res(x, y1_yaw, "真値 yaw")

    y2_roll = [y1_roll[0]]
    y2_pitch = [y1_pitch[0]]
    y2_yaw = [y1_yaw[0]]
    for i, data in enumerate(rbf_mran._test_pre_res) :
        y2_roll.append(y2_roll[i] + data[0])
        y2_pitch.append(y2_pitch[i] + data[1])
        y2_yaw.append(y2_yaw[i] + data[2])
    plot_res(x, y2_roll, "推測 roll")
    plot_res(x, y2_pitch, "推測 pitch")
    plot_res(x, y2_yaw, "推測 yaw")

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95)
    plt.ticklabel_format(style='plain',axis='y')
    plt.ticklabel_format(style='plain',axis='x')

    plt.show()