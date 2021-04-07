import numpy as np
import random
import argparse

import system

def generate_data(sys_type, train_file, val_file):
    """
    データ自動生成
    Returns:
        -(int):
            生成できたかどうか．できたなら0，できなかったら-1
        -(list):
            生成したファイルまでの相対パス
    """
    sys = None
    if sys_type == 'siso':
        sys = system.siso.SISO()
    elif sys_type == 'mimo':
        sys = system.mimo.MIMO()
    else :
        print('----- Gen new data failed : No such system type. -----')
        return -1
    
    data_len = 5000 # 生成されるデータファイルの行数
    for fn in [train_file, val_file] :
        with open(fn, mode='w') as f:
            sys_info = sys.get_sys_info()
            f.write(str(sys_info['info_num'])+'\n')
            f.write(str(sys_info['nx'])+'\n')
            f.write(str(sys_info['nu'])+'\n')
            for n in range(data_len):
                x = sys.get_x(n)
                u = sys.get_u()
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
    a, b = generate_data(args.sys, train_file, val_file)