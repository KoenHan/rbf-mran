import os
import numpy as np
import random
import argparse
import yaml

import system

def gen_param_file_from_cmd(param_file):
    """
    コマンドラインの入力からパラメータファイル生成
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
        print('Enter '+p[0]+' value.(Default: '+str(p[1])+')')
        tmp = input('>> ')
        if tmp : param[p[0]] = type(p[1])(tmp)

    with open(param_file, 'w') as f:
        yaml.dump(param, f, default_flow_style=False)
    print('Save as param file: ', param_file)

def gen_data(sys_type, train_file, val_file, data_len):
    """
    データ自動生成
    Returns:
        -(int):
            生成できたかどうか．できたなら0，できなかったら-1
    """
    sys = None
    if sys_type == 'siso':
        sys = system.siso.SISO()
    elif sys_type == 'mimo':
        sys = system.mimo.MIMO()
    else :
        print('Generate new data failed: No such system type.\nGot system type: '+sys_type)
        return -1
    
    for fn in [train_file, val_file] :
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

    return 0
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='system type(siso or mimo)', default='siso')
    parser.add_argument('-tf', '--train_file', default='train.txt')
    parser.add_argument('-vf', '--val_file', default='val.txt')
    args = parser.parse_args()

    train_file = './data/'+args.sys+'/'+args.train_file
    val_file = './data/'+args.sys+'/'+args.val_file
    a = gen_data(args.sys, train_file, val_file)