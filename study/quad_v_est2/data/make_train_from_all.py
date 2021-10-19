if __name__ == "__main__" :
    datas = []
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    with open("train.txt", "w") as f:
        for i, data in enumerate(datas) :
            tmp = "\t".join(data[-7:])
            f.write(tmp+"\n")
