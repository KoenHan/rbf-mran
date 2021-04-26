import os
import numpy as np
import random
import argparse
import yaml

import system
from utils import save_param

def gen_param_file_from_cmd(param_file):
    """
    コマンドラインの入力からパラメータファイル生成．
    """
    param = {
        'past_sys_input_num': 1,
        'past_sys_output_num': 1,
        'init_h': 0,
        'E1': 0.01,
        'E2': 0.01,
        'E3': -1,
        'E3_max': 1.2,
        'E3_min': 0.6,
        'gamma': 0.997,
        'Nw': 48,
        'Sw': 48
    }

    for p in param.items():
        if p[0] == 'init_h' or p[0] == 'E3' : continue
        print('Enter '+p[0]+' testue.(Default: '+str(p[1])+')')
        tmp = input('>> ')
        if tmp : param[p[0]] = type(p[1])(tmp)

    save_param(param, param_file)

def gen_data(sys_type, train_file, test_file, data_len, ncx=-1, ncu=-1):
    """
    データ自動生成
    Params:
        sys_type(str):
            システムのタイプ．
            sisoかmimoのみ可でそれ以外の場合は何も生成しない．
        train_file(str):
            学習用データ保存先．
        test_file(str):
            検証用データ保存先．
        data_len(int):
            生成される学習/検証用データの長さ（＝行数）．
        ncx(int):
            n_change_xの略．
            xの生成式を切り替える境目．
            -1ならinfと解釈．
        ncu(int):
            n_change_uの略．
            uの生成式を切り替える境目．
            -1ならinfと解釈．
        
    Returns:
        -(int):
            生成できたかどうか．
            できたなら0，できなかったら-1．
    """
    sys = None
    if sys_type == 'siso':
        sys = system.siso.SISO(ncx=ncx, ncu=ncu)
    elif sys_type == 'mimo':
        sys = system.mimo.MIMO()
    else :
        print('Generate new data failed: No such system type.\nGot system type: '+sys_type)
        return -1
    
    for fn in [train_file, test_file] :
        fpath = os.path.dirname(fn)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        with open(fn, mode='w') as f:
            sys_info = sys.get_sys_info()
            f.write(str(sys_info['info_num'])+'\n')
            f.write(str(sys_info['nx'])+'\n')
            f.write(str(sys_info['nu'])+'\n')
            for n in range(data_len):
                x = sys.get_x(n)
                u = sys.get_u(n)
                s = sys.gen_data_list(x, u)
                sys.set_pre_x_and_u(x, u)

                f.write('\t'.join(map(str, s))+'\n')

    print('Generated new train/test data.')
    return 0
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='system type(siso or mimo)', default='siso')
    parser.add_argument('-tr', '--train_file', default='train.txt')
    parser.add_argument('-te', '--test_file', default='test.txt')
    args = parser.parse_args()

    train_file = './data/'+args.sys+'/'+args.train_file
    test_file = './data/'+args.sys+'/'+args.test_file
    a = gen_data(args.sys, train_file, test_file)