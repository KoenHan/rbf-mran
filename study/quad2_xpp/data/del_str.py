def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

def make_file(infile, outfile) :
    print('start ', infile)
    with open(infile, "r") as ifile, open(outfile, "w") as ofile :
        while True :
            line = ifile.readline()
            if line == 'end\n' :
                break
            line = line.split()
            if line == [] :
                continue
            elif not is_num(line[0]) :
                continue
            elif len(line) == 2 :
                ofile.write(' '.join(list(map(str, line))) + '\n')
    print('end ', infile)


if __name__=='__main__' :
    make_file('train_def.txt', 'train.txt')