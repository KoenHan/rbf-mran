# 見やすさのために結果をcheck_data_result.txtに保存するときがある

import numpy as np
import os
import argparse
import pandas as pd

pd.options.display.float_format = '{:.6f}'.format

def check(file):
    with open(file, mode='r') as f:
        data = [list(map(float, s.strip().split())) for s in f.readlines()]

    ax_name = ['rollrate', 'pitchrate', 'yawrate']

    ny = int(data[1][0])
    data = data[int(data[0][0]) + 1:]
    for d_ax in range(ny):
        y = [d[d_ax] for d in data]
        y_upper = []
        y_lower = []
        for item in y :
            if item > 0 : y_upper.append(item)
            else : y_lower.append(item)
        print()
        print(ax_name[d_ax])
        df_all = pd.DataFrame({'all': y})
        df_upper = pd.DataFrame({'upper': y_upper})
        df_lower = pd.DataFrame({'lower': y_lower})
        print(df_all.describe())
        print(df_upper.describe())
        print(df_lower.describe())

if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-sn', '--study_name', required=True)
    args = parser.parse_args()

    train_file = f'./study/{args.study_name}/data/train.txt'

    check(train_file)