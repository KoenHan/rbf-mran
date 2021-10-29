'''
データファイルを指定行数に削減する
'''

if __name__ == "__main__" :
    start = 0
    start += 3 # 先頭の3行は必ずいるので
    target_line_num = 400000
    end = start + target_line_num
    cnt = 0
    with open("train_def.txt", "r") as infile, open("train.txt", "w") as outfile :
        while cnt < end :
            line = infile.readline()
            if cnt < 3 or cnt >= start :
                outfile.write(line)
            cnt += 1
