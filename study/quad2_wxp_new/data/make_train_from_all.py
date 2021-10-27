if __name__ == "__main__" :
    GAIN = 1e4
    datas = []
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    '''
    出力ファイルの一行の中身
    wxp_t w_t-1 u_t
    '''
    pre_wx = 0
    with open("train.txt", "w") as f:
        f.write('2\n')
        f.write('1\n') # wxp
        f.write('2\n') # 下のwとu
        for idx, data in enumerate(datas) :
            if idx == 0:
                pre_wx = float(data[4])
                continue
            wx = float(data[4])
            wy = float(data[5])
            wz = float(data[6])
            wxp = wx - pre_wx
            w = wy*wz
            # rps_cw1 = GAIN*float(data[-4])
            # rps_cw2 = GAIN*float(data[-3])
            rps_ccw1 = GAIN*float(data[-2])
            rps_ccw2 = GAIN*float(data[-1])
            u = rps_ccw1*abs(rps_ccw1) - rps_ccw2*abs(rps_ccw2)
            tmp = "\t".join([str(wxp), str(pre_wx), str(u)])
            f.write(tmp+"\n")
            pre_wx = wx
