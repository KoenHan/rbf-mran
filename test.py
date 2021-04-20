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
    print(p[0])
    print(type(p[1]))
    print(type(p[1])(1.0))