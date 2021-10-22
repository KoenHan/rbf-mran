import random

def sys_linear(x) :
    return 2*x + 0.5

def gen_file(fname) :
    data_len = 50000
    x_data = []
    y_data = []
    for i in range(data_len) :
        x = random.uniform(-1, 1)
        x_data.append(x)
        y_data.append(sys_linear(x))

    with open(fname, 'w') as f:
        f.write('2\n1\n1\n')
        for y, x in zip(y_data, x_data) :
            f.write(f'{y}\t{x}\n')

def main() :
    gen_file('train.txt')
    gen_file('test.txt')

if __name__ == "__main__" :
    main()