import yaml
import os
import numpy as np

def save_ndarray(fh, array):
    np.savetxt(fh, array, fmt='% .18e', delimiter = "\t")
    fh.write("\n")

def save_param(param, param_file):
    with open(param_file, 'w') as f:
        yaml.dump(param, f, default_flow_style=False)
    print('Save as param file: ', param_file)

def load_param(param_file):
    with open(param_file) as f:
        config = yaml.safe_load(f)
    print('Loaded param file:', param_file)
    return config

def gen_study(study_name):
    study_folder = './study/'+study_name
    for directory in ['/data', '/model', '/history']:
        fpath = study_folder+directory
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
    return study_folder

def save_args(args, file):
    with open(file, 'w') as f :
        yaml.dump(vars(args), f, default_flow_style=False)