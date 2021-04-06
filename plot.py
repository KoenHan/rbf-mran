import numpy as np
import matplotlib.pyplot as plt

def plot_var_res(val_file, pre_res_file):
    with open(val_file, mode='r') as f:
        data1 = [s.strip().split('\t') for s in f.readlines()]
    with open(pre_res_file, mode='r') as f:
        data2 = [s.strip().split('\t') for s in f.readlines()]
    
    x = [i for i in range(len(data1))]
    y1 = [float(d[0]) for d in data1]
    y2 = [float(d[0]) for d in data2]
    # y2の方は予測結果なので，nyだけ遅れるので適宜埋める
    y2 = [np.nan for _ in range(len(y1) -len(y2))] + y2

    ps = 1600 # plot_start
    pl = 300 # plot_len
    fig = plt.figure(figsize=(15, 15))
    for ax_pos in range(311, 314) :
        ax = fig.add_subplot(ax_pos)
        plt.plot(x[ps:ps+pl], y1[ps:ps+pl], label="実測値 val")
        plt.plot(x[ps:ps+pl], y2[ps:ps+pl], label="推測値 pre_res")
        ax.grid()
        ps += pl
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.legend()

def plot_err_hist(err_hist_file, plot_type = 0):
    with open(err_hist_file, mode='r') as f:
        data = [float(s.strip()) for s in f.readlines()]

    Nw = int(data[0])
    y = [np.nan for _ in range(Nw)] + data[1:] # Nwだけ遅れるので適宜埋める
    if plot_type == 0 : 
        """
        先頭から一定長さだけプロット
        """
        x = [i for i in range(len(y))]

        plot_size = 200
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(311)
        plt.plot(x[:plot_size], y[:plot_size], label="誤差 Id")
        ax = fig.add_subplot(312)
        plt.plot(x[plot_size:2*plot_size], y[plot_size:2*plot_size], label="誤差 Id")
        ax = fig.add_subplot(313)
        plt.plot(x[2*plot_size:3*plot_size], y[2*plot_size:3*plot_size], label="誤差 Id")
    elif plot_type == 1 :
        """
        全部プロット
        """
        x = [i for i in range(len(y))]
        fig = plt.figure(figsize=(15, 15))
        plt.plot(x, y, label="誤差 Id")
    else :
        """
        一定間隔ごとプロット
        """
        gap = 100
        x = [i for i in range(0, len(y), gap)]
        y = y[::gap]

        fig = plt.figure(figsize=(15, 3))
        plt.plot(x, y, label="誤差 Id")
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.legend()
    
def plot_h(h_hist_file):
    with open(h_hist_file, mode='r') as f:
        data = [int(s.strip()) for s in f.readlines()]

    x = [i for i in range(len(data))]
    y = data

    plot_size = 200
    fig = plt.figure(figsize=(15, 15))
    plt.plot(x, y, label="隠れニューロン数 h")
    plt.grid()
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.legend()

if __name__ == "__main__":
    plot_var_res('./data/val.txt', './data/pre_res.txt')
    plot_err_hist('./model/history/error.txt', plot_type=1)
    plot_h('./model/history/h.txt')
    plt.show()