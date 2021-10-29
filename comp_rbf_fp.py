import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def make_file(infile, outfile) :
    print('start ', infile)
    with open(infile, "r") as ifile, open(outfile, "w") as ofile :
        while True :
            line = ifile.readline()
            if line == 'end' :
                break
            line = line.split()
            if line == [] :
                continue
            elif not is_num(line[0]) :
                continue
            elif len(line) == 4 :
                ofile.write(' '.join(list(map(str, line))) + '\n')
    print('end ', infile)

if __name__ == "__main__" :
    make_file('/home/han/catkin_ws/tmp.txt', 'rbfo.txt') # tmp.txtは上書きされたのでもうない
    make_file('/home/han/catkin_ws/tmp2.txt', 'fpo.txt')
    rbf_file = 'rbfo.txt' # tmp.txt
    fp_file = 'fpo.txt' # tmp2.txt
    ww = []
    uuuu = []
    rbfo = []
    # fpo = []
    with open(rbf_file, mode='r') as f:
        l = f.readlines()
    for s in l :
        data = list(map(float, s.strip().split()))
        # if len(data) < 4 :
        #     continue
        rbfo.append(data[0])
        # fpo.append(data[1])
        ww.append(data[2])
        uuuu.append(data[3])

    ww2 = []
    uuuu2 = []
    fpo = []
    with open(fp_file, mode='r') as f:
        l = f.readlines()
    for s in l :
        data = list(map(float, s.strip().split()))
        # if len(data) < 4 :
        #     continue
        fpo.append(data[0])
        ww2.append(data[2])
        uuuu2.append(data[3])

    # 0.0043557267636060715 -0.0057880156673491
    # 346.45977783203125 -327.7222900390625

    # '''
    fig = plt.figure(figsize = (8, 8))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')

    # Axesのタイトルを設定
    # ax.set_title("rbf", size = 20)
    ax.set_xlabel("ww", size = 14)
    ax.set_ylabel("uuuu", size = 14)
    ax.set_zlabel("op", size = 14)

    # 軸目盛を設定
    # ax.set_xticks(list(range(-0.006, 0.0045, 0.0005))
    # ax.set_yticks(list(range(-330, 350, 10))
    # ax.set_zticks(list(range(-330, 350, 10))

    LEN = 10000
    ax.scatter(ww[:LEN], uuuu[:LEN], rbfo[:LEN], color = "red", s=0.5)
    ax.scatter(ww2[:LEN], uuuu2[:LEN], fpo[:LEN], color = "blue", s=0.5)

    plt.show()
    # '''