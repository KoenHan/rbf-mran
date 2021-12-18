import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

LINEWIDTH = 0.8
WINWIDTH = 15
WINHEIGHT = 15

def plot_pre_res(gt_file, pre_res_file, plot_start, plot_len, title, fig_folder='./fig/'):
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

    # ax_name = ['x', 'y', 'z', 'w', 'rollrate', 'pitchrate', 'yawrate']
    # ax_name = ['rollrate', 'pitchrate', 'yawrate']
    ax_name = ['q0rate', 'q1rate', 'q2rate', 'q3rate']

    ny = int(data1[1][0])
    data1 = data1[int(data1[0][0]) + 1:]
    x = [i for i in range(len(data1))]
    for d_ax in range(ny):
        y1 = [d[d_ax] for d in data1]
        y2 = [d[d_ax] for d in data2]

        # -0.01中心に1.9倍してみたが，完璧には重ならなかったし，
        # 多分1.9以外にしても全ての方向できれいに重なることはなさそう
        # for yi in range(len(y2)) :
        #     y2[yi] = 1.9 *(y2[yi] + 0.01) - 0.01
        # y2の方は予測結果なので，past_sys_input_numだけ遅れるので適宜埋める
        y2 = [np.nan for _ in range(len(y1) - len(y2))] + y2

        fig = plt.figure(title+'_x'+str(d_ax), figsize=(WINWIDTH, WINHEIGHT))
        ps = plot_start
        pl = plot_len
        # '''
        for ax_pos in range(311, 314) :
            ax = fig.add_subplot(ax_pos, xticks=[i for i in range(ps, ps + pl + 1, 100)])
            plt.plot(x[ps:ps+pl], y1[ps:ps+pl], linewidth=LINEWIDTH, label="実測値 ")#+os.path.basename(gt_file))
            plt.plot(x[ps:ps+pl], y2[ps:ps+pl], linewidth=LINEWIDTH, label="推定値 ")#+os.path.basename(pre_res_file))
            ax.grid()
            ps += pl
        plt.title(ax_name[d_ax], y=-0.2)
        plt.savefig(fig_folder+ax_name[d_ax]+'.png')
        plt.legend()
        # '''
        ''' レジュメ用フォーマット
        plt.xticks([i for i in range(ps, ps + pl + 1, 100)])
        plt.plot(x[ps:ps+pl], y1[ps:ps+pl], label="実測値", linewidth=LINEWIDTH)
        plt.plot(x[ps:ps+pl], y2[ps:ps+pl], label="推定値", linewidth=LINEWIDTH)
        if d_ax == 3 :
            plt.ylim(0.9975, 1.00005)
        plt.grid()
        plt.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.99)
        plt.legend()
        plt.title(ax_name[d_ax], y=-0.2)
        '''

def plot_res_err(gt_file, pre_res_file, title, fig_folder='./fig/'):
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
        y2 = [np.nan for _ in range(len(y1) - len(y2))] + y2
        y = []
        error_type = '_RE'
        try :
            y = [abs(yy2 - yy1)/yy1 for yy1, yy2 in zip(y1, y2)] # 誤差率
        except ZeroDivisionError :
            print('Detected zero division error!!! Use absolute error.')
            y = [yy2 - yy1 for yy1, yy2 in zip(y1, y2)] # 誤差
            error_type = '_AE'
        fig_title = title+'_te_'+str(d_ax)+error_type
        fig = plt.figure(fig_title, figsize=(WINWIDTH, WINHEIGHT))

        len_y = len(y)
        x = [i for i in range(len_y)]
        plt.plot(x, y, label=title+error_type, linewidth=LINEWIDTH)

        plt.savefig(fig_folder+fig_title+'.png')
        plt.legend()

def plot_err_hist(err_hist_file, mode=0, fig_folder='./fig/'):
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
        fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
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
        len_y = len(y)
        x = [i for i in range(len_y)]
        fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
        # plt.xticks([i for i in range(0, len_y + 1, 300)])
        plt.plot(x, y, label=title, linewidth=LINEWIDTH)
    else :
        """
        一定間隔ごとプロット
        """
        gap = 100
        x = [i for i in range(0, len(y), gap)]
        y = y[::gap]

        fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
        plt.plot(x, y, label=title)
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    # plt.legend()
    # plt.title(title, y=-0.25)
    plt.savefig(fig_folder+'id_hist.png')

def plot_h(h_hist_file, fig_folder='./fig/'):
    with open(h_hist_file, mode='r') as f:
        data = [int(s.strip()) for s in f.readlines()]

    len_data = len(data)
    x = [i for i in range(len_data)]
    y = data

    title = "隠れニューロン数 h"
    plot_size = 200
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
    # plt.xticks([i for i in range(0, len_data + 1, 300)])
    # plt.yticks([i for i in range(0, max(data)+1, 1)])
    plt.plot(x, y, label=title, linewidth=LINEWIDTH)
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    # plt.legend()
    # plt.title(title, y=-0.25)
    plt.savefig(fig_folder+'h_hist.png')

def plot_all(err_file, h_hist_file, test_file, pre_res_file, plot_start, plot_len, mode=1):
    plot_pre_res(test_file, pre_res_file, plot_start, plot_len)
    plot_err_hist(err_file, mode=mode)
    plot_h(h_hist_file)
    plt.show()

def plot_study(study_name, plot_start, plot_len, eh_mode=1):
    project_folder = './study/'+study_name
    test_file       = project_folder+'/data/test.txt'
    test_ps_file    = project_folder+'/data/test_pre_res.txt'
    train_file      = project_folder+'/data/train.txt'
    train_ps_file   = project_folder+'/data/train_pre_res.txt'
    h_hist_file     = project_folder+'/history/h.txt'
    err_file        = project_folder+'/history/error.txt'
    fig_folder      = project_folder+'/fig/'
    if not os.path.isdir(fig_folder) :
        os.makedirs(fig_folder)
    if os.path.isfile(test_ps_file) :
        plot_pre_res(test_file, test_ps_file, plot_start, plot_len, 'test', fig_folder)
        plot_res_err(test_file, test_ps_file, 'test_err', fig_folder)
    if os.path.isfile(train_ps_file):
        plot_pre_res(train_file, train_ps_file, plot_start, plot_len, 'realtime', fig_folder)
        plot_res_err(train_file, train_ps_file, 'train_err', fig_folder)
    plot_err_hist(err_file, eh_mode, fig_folder)
    plot_h(h_hist_file, fig_folder)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--study_name', required=True)
    parser.add_argument('-ps', '--plot_start', type=int, default=4300)
    parser.add_argument('-pl', '--plot_len', type=int, default=500)
    parser.add_argument('-m', '--mode', type=int, default=1)
    args = parser.parse_args()

    plot_study(args.study_name, args.plot_start, args.plot_len, eh_mode=args.mode)

