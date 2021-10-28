

class QuadWxp :
    def __init__(self) :
        U_G = 73.0
        U_DIFF_LIM = 61.0
        MAX_RPS    = U_G + U_DIFF_LIM

        self.dt = 0.01
        self.A_OF_GRAVITY = 9.80
        self.VAR_SIZE = 13
        self.I_XX = 0.007
        self.I_YY = 0.007
        self.I_ZZ = 0.012
        self.ROTOR_DISTANCE = 0.34
        self.MASS_OF_MACHINE = 0.716
        self.MAX_RPS_POW = MAX_RPS**2
        self.MAX_THRUST = self.MASS_OF_MACHINE*self.A_OF_GRAVITY*self.MAX_RPS_POW/(4*U_G*U_G)
        self.TORQUE_RATE = 0.25/MAX_RPS/MAX_RPS
        # self.RADIAN = 3.1415
        # self.TWICE_RADIAN = 2*self.RADIAN

        # self.u = [0 for _ in range(4)]
        # self.var_p = [0 for _ in range(self.VAR_SIZE)]
        # self.var_and_z_i = [0 for _ in range(self.VAR_SIZE)]
        # self.var_and_z_i[0] = 1.0 # クオータニオンだから

        # self.GAIN = 1e4

    def wxp_f(self, ww, uuuu) :
        wyp = -0.5*(2.0*(self.I_XX-self.I_ZZ)*ww+self.ROTOR_DISTANCE*self.MAX_THRUST*uuuu/self.MAX_RPS_POW)/self.I_YY
        return wyp

if __name__ == "__main__" :
    GAIN = 1e4
    datas = []

    quadwxp = QuadWxp()
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    '''
    出力ファイルの一行の中身
    wxp_t w_t-1 u_t
    '''
    pre_wx = 0
    pre_ww = 0
    with open("train.txt", "w") as f:
        f.write('2\n')
        f.write('1\n') # wxp
        f.write('2\n') # 下のwwとuuuu
        for idx, data in enumerate(datas) :
            wx = float(data[4])
            # wy = float(data[5])
            wz = float(data[6])
            if idx == 0:
                pre_ww = wx*wz
                continue
            rps_cw1 = GAIN*float(data[-4])
            rps_cw2 = GAIN*float(data[-3])
            # rps_ccw1 = GAIN*float(data[-2])
            # rps_ccw2 = GAIN*float(data[-1])
            uuuu = rps_cw1*abs(rps_cw1) - rps_cw2*abs(rps_cw2) # 論文では下になってるのになぜかこれになってた
            # uuuu = rps_ccw1**2 - rps_ccw2**2
            wyp = quadwxp.wxp_f(pre_ww, uuuu)
            # print(org_wxp, wxp)
            tmp = "\t".join([str(wyp), str(pre_ww), str(uuuu)])
            f.write(tmp+"\n")
            # pre_wx = wx
            pre_ww = wx*wz
