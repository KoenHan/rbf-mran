import argparse
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from RBF_MRAN import RBF_MRAN
from generate_data import *
from plot import plot_study
from utils import load_param, gen_study, save_args, save_ndarray

LINEWIDTH = 0.8
WINWIDTH = 15
WINHEIGHT = 15

def plot_res(x, y, label):
    # plt.xticks([i for i in range(0, len_data + 1, 300)])
    # plt.yticks([i for i in range(0, max(data)+1, 1)])
    plt.plot(x, y, label=label, linewidth=LINEWIDTH)
    plt.grid()
    plt.legend()
    # plt.title(title, y=-0.25)
    # plt.savefig(fig_folder+'/h_hist.png')

def get_rbf_mran_and_hist_len(study_folder) :
    param_file = study_folder+'/model/param.yaml'
    train_file = study_folder+'/data/train.txt'
    with open(train_file, mode='r') as f:
        l = f.readlines()
    train_datas = [list(map(float, s.strip().split())) for s in l]

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
        use_exist_net=True,
        readonly=True) # 既存のネットワークを使うかどうか
    return rbf_mran, hist_len

def quat(start) :
    study_folder = "./study/ros_test_angle_quat"
    rbf_mran, hist_len = get_rbf_mran_and_hist_len(study_folder)

    qrs_file = study_folder+'/data/quat_rate_sysin.txt'
    with open(qrs_file, mode='r') as f:
        l = f.readlines()
    qrs_datas = [list(map(float, s.strip().split())) for s in l]

    idx = int(qrs_datas[0][0]) + start
    horizen = 50
    y = qrs_datas[idx-hist_len:idx+horizen]
    for data in y[:-1] : # 加速度を学習したので長さはhorizen-1でいい
        rbf_mran.test(data[4:7] + data[-4:])

    title = "モデル予測結果(姿勢)"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))

    x = [i for i in range(start, start+horizen)]
    y1_q0 = []
    y1_q1 = []
    y1_q2 = []
    y1_q3 = []
    for data in y[hist_len:hist_len+horizen] :
        y1_q0.append(data[0])
        y1_q1.append(data[1])
        y1_q2.append(data[2])
        y1_q3.append(data[3])


    y2_q0 = [y1_q0[0]]
    y2_q1 = [y1_q1[0]]
    y2_q2 = [y1_q2[0]]
    y2_q3 = [y1_q3[0]]
    dt = 1/50
    # todo: 角速度をそのまま積分しても角度にならないので修正する
    # https://www.kazetest.com/vcmemo/quaternion/quaternion.htm
    for data in rbf_mran._test_pre_res :
        y2_q0.append(y2_q0[-1] + data[0]*dt)
        y2_q1.append(y2_q1[-1] + data[1]*dt)
        y2_q2.append(y2_q2[-1] + data[2]*dt)
        y2_q3.append(y2_q3[-1] + data[3]*dt)

    # plot_res(x, y1_q0, "真値 q0")
    # plot_res(x, y2_q0, "推測 q0")
    plot_res(x, y1_q1, "真値 q1")
    plot_res(x, y2_q1, "推測 q1")
    # plot_res(x, y1_q2, "真値 q2")
    # plot_res(x, y2_q2, "推測 q2")
    # plot_res(x, y1_q3, "真値 q3")
    # plot_res(x, y2_q3, "推測 q3")

    # print(rbf_mran._test_pre_res)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95)
    plt.ticklabel_format(style='plain',axis='y')
    plt.ticklabel_format(style='plain',axis='x')
    plt.show()


def euler(start) :
    study_folder = "./study/ros_test_angle_euler3"
    rbf_mran, hist_len = get_rbf_mran_and_hist_len(study_folder)
    rbf_mran2 = deepcopy(rbf_mran)

    qrs_file = study_folder+'/data/quat_rate_sysin.txt'
    with open(qrs_file, mode='r') as f:
        l = f.readlines()
    qrs_datas = [list(map(float, s.strip().split())) for s in l]


    idx = int(qrs_datas[0][0]) + start
    horizen = 50
    y = qrs_datas[idx:idx+horizen]
    w_euler = []
    w_euler2 = []
    x = [i for i in range(start, start+horizen)]
    print('x len ', len(x))
    for i, data in enumerate(y) :
        if len(rbf_mran._test_pre_res) == 0 :
            w_euler = deepcopy(data[4:7])
        else :
            w_euler = deepcopy(rbf_mran._test_pre_res[-1]).tolist() # 実際の制御ではこうなるから
        w_euler2 = deepcopy(data[4:7])
        print(f'x {x[i]}')
        print('wの推測値を使う方：')
        print('w_euler ', w_euler)
        rbf_mran.test(deepcopy(w_euler) + deepcopy(data[-4:]))
        print('wの真値を使う方：')
        print('w_euler2 ', w_euler2)
        rbf_mran2.test(deepcopy(w_euler2) + deepcopy(data[-4:]))
        print()
        # print(type(w_euler))
        # print(type(data[-4:]))
        # print(rbf_mran._test_pre_res[-1].tolist())

    title = "モデル予測結果(姿勢)"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))

    y1_q0 = []
    y1_q1 = []
    y1_q2 = []
    y1_q3 = []
    y1_w0 = []
    y1_w1 = []
    y1_w2 = []
    for data in y :
        y1_q0.append(data[0])
        y1_q1.append(data[1])
        y1_q2.append(data[2])
        y1_q3.append(data[3])
        y1_w0.append(data[4])
        y1_w1.append(data[5])
        y1_w2.append(data[6])
    print('y1_w2 len ', len(y1_w2))
    # plot_res(x, y1_q0, "真値 q0")
    # plot_res(x, y1_q1, "真値 q1")
    # plot_res(x, y1_q2, "真値 q2")
    # plot_res(x, y1_q3, "真値 q3")
    plot_res(x, y1_w0, "真値 w0")
    # plot_res(x, y1_w1, "真値 w1")
    # plot_res(x, y1_w2, "真値 w2")

    y2_q0 = [y1_q0[0]]
    y2_q1 = [y1_q1[0]]
    y2_q2 = [y1_q2[0]]
    y2_q3 = [y1_q3[0]]
    y2_w0 = [y1_w0[0]]
    y2_w1 = [y1_w1[0]]
    y2_w2 = [y1_w2[0]]
    y3_w0 = [y1_w0[0]]
    y3_w1 = [y1_w1[0]]
    y3_w2 = [y1_w2[0]]
    dt = 1/50
    # 角速度→クオータニオン形式変換参考
    # https://www.kazetest.com/vcmemo/quaternion/quaternion.htm
    for data in rbf_mran._test_pre_res :
        gx = data[0]*dt/2.0
        gy = data[1]*dt/2.0
        gz = data[2]*dt/2.0
        # # 移動体座標系
        dq0 = -y2_q1[-1]*gx - y2_q2[-1]*gy - y2_q3[-1]*gz
        dq1 =  y2_q0[-1]*gx + y2_q2[-1]*gz - y2_q3[-1]*gy
        dq2 =  y2_q0[-1]*gy - y2_q1[-1]*gz + y2_q3[-1]*gx
        dq3 =  y2_q0[-1]*gz + y2_q1[-1]*gy - y2_q2[-1]*gx
        y2_q0.append(y2_q0[-1] + dq0)
        y2_q1.append(y2_q1[-1] + dq1)
        y2_q2.append(y2_q2[-1] + dq2)
        y2_q3.append(y2_q3[-1] + dq3)
        y2_w0.append(data[0])
        y2_w1.append(data[1])
        y2_w2.append(data[2])
    for data in rbf_mran2._test_pre_res :
        y3_w0.append(data[0])
        y3_w1.append(data[1])
        y3_w2.append(data[2])
    # plot_res(x, y2_q0, "推測 q0")
    # plot_res(x, y2_q1, "推測 q1")
    # plot_res(x, y2_q2, "推測 q2")
    # plot_res(x, y2_q3, "推測 q3")
    plot_res(x, y2_w0, "推測* w0")
    plot_res(x, y3_w0, "推測 w0")
    # plot_res(x, y2_w1, "推測 w1")
    # plot_res(x, y2_w2, "推測 w2")

    # print(rbf_mran._test_pre_res)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95)
    plt.ticklabel_format(style='plain',axis='y')
    plt.ticklabel_format(style='plain',axis='x')
    plt.show()


def position(start) :
    study_folder = "./study/ros_test_pos2"
    rbf_mran, hist_len = get_rbf_mran_and_hist_len(study_folder)

    qrs_file = study_folder+'/data/quat_rate_sysin.txt'
    with open(qrs_file, mode='r') as f:
        l = f.readlines()
    qrs_datas = [list(map(float, s.strip().split())) for s in l]

    idx = int(qrs_datas[0][0]) + start
    horizen = 50
    y = qrs_datas[idx-hist_len:idx+horizen]
    v = y[0][-7:-4]
    for data in y[:-1] : # 速度を学習したので長さはhorizen-1でいい
        rbf_mran.test(v + data[-4:])
        if len(rbf_mran._test_pre_res) == 0 :
            v = data[-7:-4]
        else :
            v = rbf_mran._test_pre_res[-1].tolist() # 実際の制御ではこうなるから

    title = "モデル予測結果(位置)"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))

    x = [i for i in range(start, start+horizen)]
    y1_x = []
    y1_y = []
    y1_z = []
    y1_xp = []
    y1_yp = []
    y1_zp = []
    for data in y[hist_len:hist_len+horizen] :
        y1_x.append(data[7])
        y1_y.append(data[8])
        y1_z.append(data[9])
        y1_xp.append(data[10])
        y1_yp.append(data[11])
        y1_zp.append(data[12])
    # plot_res(x, y1_x, "真値 x")
    # plot_res(x, y1_y, "真値 y")
    # plot_res(x, y1_z, "真値 z")
    plot_res(x, y1_xp, "真値 xp")
    plot_res(x, y1_yp, "真値 yp")
    plot_res(x, y1_zp, "真値 zp")

    y2_x = [y1_x[0]]
    y2_y = [y1_y[0]]
    y2_z = [y1_z[0]]
    y2_xp = [y1_xp[0]]
    y2_yp = [y1_yp[0]]
    y2_zp = [y1_zp[0]]
    dt = 1/50
    for data in rbf_mran._test_pre_res :
        y2_x.append(y2_x[-1] + data[0]*dt)
        y2_y.append(y2_y[-1] + data[1]*dt)
        y2_z.append(y2_z[-1] + data[2]*dt)
        y2_xp.append(data[0])
        y2_yp.append(data[1])
        y2_zp.append(data[2])
    # plot_res(x, y2_x, "推測 x")
    # plot_res(x, y2_y, "推測 y")
    # plot_res(x, y2_z, "推測 z")
    plot_res(x, y2_xp, "推測 xp")
    plot_res(x, y2_yp, "推測 yp")
    plot_res(x, y2_zp, "推測 zp")

    # print(rbf_mran._test_pre_res)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95)
    plt.ticklabel_format(style='plain',axis='y')
    plt.ticklabel_format(style='plain',axis='x')

    plt.show()

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', required=True)
    args = parser.parse_args()

    # starts = [55000, 60000, 65000]
    starts = [i for i in range(1000, 65001, 50)]
    if args.type == "pos" :
        for start in starts :
            position(start)
    elif args.type == "euler" :
        for start in starts :
            euler(start)
    elif args.type == "quat" :
        for start in starts :
            quat(start)
    else :
        print(f'no such type : {args.type}')