if __name__ == "__main__" :
    datas = []
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    max_val = -1
    idx = 6
    with open("train.txt", "w") as f:
        for i, data in enumerate(datas) :
            tmp = "\t".join(data[4:7] + data[-4:])
            if len(data) > 5 and max_val < float(data[idx]) :
                max_val = float(data[idx])
            f.write(tmp+"\n")

    print(max_val)