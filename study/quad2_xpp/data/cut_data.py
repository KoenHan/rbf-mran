'''
データファイルを指定行数に削減する
'''

if __name__ == "__main__" :
    target_line_num = 50000
    target_line_num += 3 # 先頭の3行は必ずいるので
    cnt = 0
    with open("train_def.txt", "r") as infile, open("train.txt", "w") as outfile :
        while cnt < target_line_num :
            line = infile.readline()
            outfile.write(line)
            cnt += 1
