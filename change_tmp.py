if __name__ == "__main__" :
    infile = "tmp.txt"
    outfile = "check_model_input.txt"
    with open(infile, "r") as ifile, open(outfile, "w") as ofile :
        while True :
            line = ifile.readline().split()
            if line == [] :
                break
            tmp = '0\t' + line[5] + '\t' + line[7] + '\n'
            ofile.write(tmp)