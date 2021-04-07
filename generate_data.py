import numpy as np
import random
import argparse

import system

def generate_data():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--sys', help='system type(siso or mimo)', default='siso')
    args = parser.parse_args()

    sys = None
    if args.sys == 'siso':
        sys = system.SISO()
        file_name = ['./data/siso/train.txt', './data/siso/val.txt']
    elif args.sys == 'mimo':
        sys = system.MIMO()
        file_name = ['./data/mimo/train.txt', './data/mimo/val.txt']
    else :
        print('----- No such system type. -----')
        return 0
    
    data_len = 5000 # 生成されるデータファイルの行数
    for fn in file_name :
        # x = sys.get_init_x()
        with open(fn, mode='w') as f:
            sys_info = sys.get_sys_info()
            f.write(str(sys_info['nx'])+'\n')
            f.write(str(sys_info['nu'])+'\n')
            for n in range(data_len):
                x = sys.get_x(n)
                u = sys.get_u()
                s = sys.gen_data_list(x, u)
                sys.set_pre_x_and_u(x, u)

                f.write('\t'.join(map(str, s))+'\n')
            
if __name__ == "__main__":
    generate_data()