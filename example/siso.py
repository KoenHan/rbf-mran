import numpy as np

from ..RBF_MRAN import RBF_MRAN
from ../

if __name__ == '__main__':
    rbf_mran = RBF_MRAN(
        nu=1, # システム入力(制御入力)の次元
        ny=1, # システム出力ベクトルの次元
        past_sys_input_num=1, # 過去のシステム入力保存数
        past_sys_output_num=1, # 過去のシステム出力保存数
        init_h=0, # スタート時の隠れニューロン数
        E1=0.01, E2=0.01, E3=1.2, Nw=48, Sw=48)

    start = time.time()
    rbf_mran.train('./data/siso/train.txt')
    duration = time.time()-start
    print('rbf_mran.train() duration[s]: ', str(duration))
    print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran.update_rbf_time)/len(rbf_mran.update_rbf_time))
    rbf_mran.save_hist('./model/history/siso/error.txt', './model/history/siso/h.txt')
    rbf_mran.val('./data/siso/val.txt')