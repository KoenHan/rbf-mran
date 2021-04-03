import numpy as np
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open('./data/train.txt', mode='r') as f:
        data = [s.strip().split('\t') for s in f.readlines()]
    
    x = [i for i in range(len(data))]
    y = [float(d[0]) for d in data]

    plot_size = 100
    fig = plt.figure(figsize=(15, 3))
    # ax = fig.add_subplot(111)
    plt.plot(x[:plot_size], y[:plot_size])
    plt.show()