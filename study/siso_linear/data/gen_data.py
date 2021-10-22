import random
import math

def sys_linear(x, pre_y) :
    # return (-1.3*pre_y - 2*x)/13
    return math.sin(-1.3*pre_y - 2*x)/13

def gen_file(fname) :
    data_len = 100000
    x_data = []
    y_data = [0]
    for i in range(data_len) :
        x = random.uniform(-1, 1)
        x_data.append(x)
        y_data.append(sys_linear(x, y_data[-1]))

    with open(fname, 'w') as f:
        f.write('2\n1\n1\n')
        for y, x in zip(y_data, x_data) :
            f.write(f'{y}\t{x}\n')

def main() :
    gen_file('train.txt')
    gen_file('test.txt')

if __name__ == "__main__" :
    main()