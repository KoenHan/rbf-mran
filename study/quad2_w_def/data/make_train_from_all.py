if __name__ == "__main__" :
    datas = []
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    with open("train.txt", "w") as f:
        f.write('2\n')
        f.write('3\n') # 角速度
        f.write('8\n') # クオータニオン+制御入力
        for i, data in enumerate(datas) :
            tmp = "\t".join(data[4:7]+data[:4]+data[-4:])
            f.write(tmp+"\n")
