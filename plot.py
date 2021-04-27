import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_pre_res(gt_file, pre_res_file, plot_start, plot_len, title):
    """
    指定区間内の真値と予測値をプロットするプログラム．
    見やすさを考慮して3段に分けてプロットする．

    Params:
        gt_file(str):
            真値ファイルのパス．
        pre_res_file(str):
            予測値ファイルのパス．
        plot_start(int):
            プロット開始点．
        plot_len(int):
            各段でプロットされる区間の長さ．
        title(str):
            ウィンドウタイトル．
    """
    with open(gt_file, mode='r') as f:
        data1 = [list(map(float, s.strip().split())) for s in f.readlines()]
    with open(pre_res_file, mode='r') as f:
        data2 = [list(map(float, s.strip().split())) for s in f.readlines()]

    ny = int(data1[1][0])
    data1 = data1[int(data1[0][0]) + 1:]
    x = [i for i in range(len(data1))]
    for d_ax in range(ny):
        y1 = [d[d_ax] for d in data1]
        y2 = [d[d_ax] for d in data2]
        # y2の方は予測結果なので，ny*past_sys_input_numだけ遅れるので適宜埋める
        y2 = [np.nan for _ in range(len(y1) - len(y2))] + y2

        fig = plt.figure(title+'_x'+str(d_ax), figsize=(15, 15))
        ps = plot_start
        pl = plot_len
        for ax_pos in range(311, 314) :
            ax = fig.add_subplot(ax_pos)
            plt.plot(x[ps:ps+pl], y1[ps:ps+pl], label="実測値 ")#+os.path.basename(gt_file))
            plt.plot(x[ps:ps+pl], y2[ps:ps+pl], label="推測値 ")#+os.path.basename(pre_res_file))
            ax.grid()
            ps += pl
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
        plt.legend()
        plt.title('軸: x'+str(d_ax), y=-0.3)

def plot_err_hist(err_hist_file, mode=0):
    with open(err_hist_file, mode='r') as f:
        data = [float(s.strip()) for s in f.readlines()]

    title = "学習中の誤差 Id"
    Nw = int(data[0])
    y = [np.nan for _ in range(Nw)] + data[1:] # Nwだけ遅れるので適宜埋める
    if mode == 0 :
        """
        先頭から一定長さだけプロット
        """
        x = [i for i in range(len(y))]

        plot_size = 200
        fig = plt.figure(title, figsize=(15, 15))
        ax = fig.add_subplot(311)
        plt.plot(x[:plot_size], y[:plot_size], label=title)
        ax = fig.add_subplot(312)
        plt.plot(x[plot_size:2*plot_size], y[plot_size:2*plot_size], label=title)
        ax = fig.add_subplot(313)
        plt.plot(x[2*plot_size:3*plot_size], y[2*plot_size:3*plot_size], label=title)
    elif mode == 1 :
        """
        全部プロット
        """
        x = [i for i in range(len(y))]
        fig = plt.figure(title, figsize=(15, 15))
        plt.plot(x, y, label=title)
    else :
        """
        一定間隔ごとプロット
        """
        gap = 100
        x = [i for i in range(0, len(y), gap)]
        y = y[::gap]

        fig = plt.figure(title, figsize=(15, 3))
        plt.plot(x, y, label=title)
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.legend()

def plot_h(h_hist_file):
    with open(h_hist_file, mode='r') as f:
        data = [int(s.strip()) for s in f.readlines()]

    x = [i for i in range(len(data))]
    y = data

    title = "隠れニューロン数 h"
    plot_size = 200
    fig = plt.figure(title, figsize=(15, 15))
    plt.plot(x, y, label=title)
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.legend()

def plot_all(err_file, h_hist_file, test_file, pre_res_file, plot_start, plot_len, mode=1):
    plot_pre_res(test_file, pre_res_file, plot_start, plot_len)
    plot_err_hist(err_file, mode=mode)
    plot_h(h_hist_file)
    plt.show()

def plot_study(study_name, plot_start, plot_len, need_rt=False, eh_mode=1):
    project_folder = './study/'+study_name
    test_file       = project_folder+'/data/test.txt'
    test_ps_file    = project_folder+'/data/test_pre_res.txt'
    train_file      = project_folder+'/data/train.txt'
    train_ps_file   = project_folder+'/data/train_pre_res.txt'
    h_hist_file     = project_folder+'/history/h.txt'
    err_file        = project_folder+'/history/error.txt'
    if not need_rt and os.path.isfile(test_ps_file) :
        plot_pre_res(test_file, test_ps_file, plot_start, plot_len, 'test')
    if need_rt and os.path.isfile(train_ps_file):
        plot_pre_res(train_file, train_ps_file, plot_start, plot_len, 'realtime')
    plot_err_hist(err_file, mode=eh_mode)
    plot_h(h_hist_file)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-ps', '--plot_start', type=int, default=4300)
    parser.add_argument('-pl', '--plot_len', type=int, default=500)
    parser.add_argument('-m', '--mode', type=int, default=1)
    args = parser.parse_args()

    plot_study(args.study_name, args.plot_start, args.plot_len, args.mode)
