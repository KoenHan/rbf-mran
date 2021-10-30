'''
予測値と真値の差を見るためのコード
'''

import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.10f}'.format

with open('rbf_fp_diff.txt', "r") as f :
    data = [list(map(float, s.strip().split())) for s in f.readlines()]

print(len(data))
diff = []
for d in data :
    tmp = d[0] - d[1]
    diff.append(tmp)
    # if tmp > 8.78 :
    #     print(d[0], d[1])

df_all = pd.DataFrame({'all': diff})

print(df_all.describe())

