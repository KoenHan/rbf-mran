import argparse
import yaml

from RBF_MRAN import RBF_MRAN
from generate_data import generate_data
from plot import plot_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys', help='Specific system type(siso or mimo)', default='siso')
    parser.add_argument('-gnd', '--gen_new_data', help='If True, generate new train/val data. Default: False.', action='store_true')
    parser.add_argument('-tf', '--train_file', default='train.txt')
    parser.add_argument('-vf', '--val_file', default='val.txt')
    parser.add_argument('-pcf', '--param_config_file', default='siso.yaml')
    args = parser.parse_args()

    train_file = './data/'+args.sys+'/'+args.train_file
    val_file = './data/'+args.sys+'/'+args.val_file
    if args.gen_new_data :
        gen_res = generate_data(args.sys, train_file, val_file)
        if gen_res < 0 :
            exit()
        print('Generated new train/val data.')

    with open(train_file, mode='r') as f:
        l = f.readlines()
    datas = [list(map(float, s.strip().split())) for s in l]

    config_file = './model/param/'+args.param_config_file
    with open(config_file) as f:
        print('Loaded param config file:', config_file)
        config = yaml.safe_load(f)

    rbf_mran = RBF_MRAN(
        nu=int(datas[2][0]), # システム入力(制御入力)の次元
        ny=int(datas[1][0]), # システム出力ベクトルの次元
        past_sys_input_num=config['past_sys_input_num'], # 過去のシステム入力保存数
        past_sys_output_num=config['past_sys_output_num'], # 過去のシステム出力保存数
        init_h=config['init_h'], # スタート時の隠れニューロン数
        E1=config['E1'],
        E2=config['E2'],
        E3=config['E3'],
        E3_max=config['E3_max'],
        E3_min=config['E3_min'],
        gamma=config['gamma'],
        Nw=config['Nw'],
        Sw=config['Sw'])
    
    # 学習
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.train(data)
    print('mean rbf_mran.update_rbf() duration[s]: ', sum(rbf_mran.update_rbf_time)/len(rbf_mran.update_rbf_time))
    print('MAE: ', rbf_mran.calc_MAE())
    
    with open(val_file, mode='r') as f:
        l = f.readlines()
    datas = [s.strip().split() for s in l]

    # 検証
    for data in datas[int(datas[0][0])+1:] :
        rbf_mran.val(data)
    
    # 色々保存とプロット
    pre_res_file = './data/'+args.sys+'/pre_res.txt'
    err_file = './model/history/'+args.sys+'/error.txt'
    h_hist_file = './model/history/'+args.sys+'/h.txt'
    rbf_mran.save_res(err_file, h_hist_file, pre_res_file)
    plot_all(
        err_file=err_file,
        h_hist_file=h_hist_file,
        val_file=val_file,
        pre_res_file=pre_res_file,
        mode=1)


