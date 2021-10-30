'''
データファイルを指定行数に削減する
'''

if __name__ == "__main__" :
    start = 2000000
    target_line_num = 1000000
    start += 3 # 先頭の3行は必ずいるので
    end = start + target_line_num + 3
    cnt = 0
    infile = "train_def.txt"
    outfile = "test.txt"
    with open(infile, "r") as ifile, open(outfile, "w") as ofile :
        while cnt < end :
            line = ifile.readline()
            if cnt < 3 or cnt >= start :
                ofile.write(line)
            cnt += 1