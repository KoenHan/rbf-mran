import yaml

def save_param(param, param_file):
    with open(param_file, 'w') as f:
        yaml.dump(param, f, default_flow_style=False)
    print('Save as param file: ', param_file)