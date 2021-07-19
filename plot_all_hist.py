import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


LINEWIDTH = 0.8
WINWIDTH = 15
WINHEIGHT = 15

def plot_h(h_hist_file):
    with open(h_hist_file, mode='r') as f:
        data = [int(s.strip()) for s in f.readlines()]

    len_data = len(data)
    x = [i for i in range(len_data)]
    y = data

    # plt.xticks([i for i in range(0, len_data + 1, 300)])
    # plt.yticks([i for i in range(0, max(data)+1, 1)])
    plt.plot(x, y, label=h_hist_file, linewidth=LINEWIDTH)
    plt.grid()
    plt.legend()
    # plt.title(title, y=-0.25)
    # plt.savefig(fig_folder+'/h_hist.png')


def plot_err_hist(err_hist_file, mode=1):
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
        # fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
        # plt.xticks([i for i in range(0, len_y + 1, 300)])
        plt.plot(x, y, label=err_hist_file, linewidth=LINEWIDTH)
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
    plt.legend()
    # plt.title(title, y=-0.25)
    # plt.savefig(fig_folder+'id_hist.png')


if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--study_name', required=True)
    args = parser.parse_args()

    title = "隠れニューロン数 h"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))

    num = [i for i in range(0, 101, 5)]
    # num += [50, 80]
    for n in num :
        fpath = f'./study/{args.study_name}/history/h_{n}.txt'
        if os.path.isfile(fpath) :
            plot_h(fpath)

    title = "学習中の誤差 Id"
    fig = plt.figure(title, figsize=(WINWIDTH, WINHEIGHT))
    for n in num :
        fpath = f'./study/{args.study_name}/history/error_{n}.txt'
        if os.path.isfile(fpath) :
            plot_err_hist(fpath)

    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.show()