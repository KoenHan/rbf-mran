import argparse

'''
データファイルを指定行数に削減する
'''

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-cs', '--cut_start', type=int, required=True)
    parser.add_argument('-cl', '--cut_len', type=int, required=True)
    args = parser.parse_args()

    start = args.cut_start
    start += 3 # 先頭の3行は必ずいるので
    target_line_num = args.cut_len
    end = start + target_line_num
    cnt = 0
    with open("train_def.txt", "r") as infile, open(args.output, "w") as outfile :
        while cnt < end :
            line = infile.readline()
            if cnt < 3 or cnt >= start :
                outfile.write(line)
            cnt += 1
