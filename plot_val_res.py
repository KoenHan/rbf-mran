import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('./data/val.txt', mode='r') as f:
        data1 = [s.strip().split('\t') for s in f.readlines()]
    with open('./data/val_res.txt', mode='r') as f:
        data2 = [s.strip().split('\t') for s in f.readlines()]
    
    x = [i for i in range(len(data1))]
    y1 = [float(d[0]) for d in data1]
    y2 = [float(d[0]) for d in data2]
    # y2の方は予測結果なので，nyだけ遅れるので適宜埋める
    y2 = [np.nan for _ in range(len(y1) -len(y2))] + y2

    plot_size = 200
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(311)
    plt.plot(x[:plot_size], y1[:plot_size])
    plt.plot(x[:plot_size], y2[:plot_size])
    ax = fig.add_subplot(312)
    plt.plot(x[plot_size:2*plot_size], y1[plot_size:2*plot_size])
    plt.plot(x[plot_size:2*plot_size], y2[plot_size:2*plot_size])
    ax = fig.add_subplot(313)
    plt.plot(x[2*plot_size:3*plot_size], y1[2*plot_size:3*plot_size])
    plt.plot(x[2*plot_size:3*plot_size], y2[2*plot_size:3*plot_size])
    plt.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.99)
    plt.show()