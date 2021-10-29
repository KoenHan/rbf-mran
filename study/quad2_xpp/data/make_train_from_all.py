

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

    def calc_f(self, q4u4) :
        # (var_and_z_i[0]*var_and_z_i[2]+var_and_z_i[1]*var_and_z_i[3])
        # rps_ccw1*fabsf(rps_ccw1)+rps_ccw2*fabsf(rps_ccw2)+rps_cw1*fabsf(rps_cw1)+rps_cw2*fabsf(rps_cw2)
        xpp = 2.0*self.MAX_THRUST*q4u4/(self.MASS_OF_MACHINE*self.MAX_RPS_POW)
        return xpp

if __name__ == "__main__" :
    GAIN = 1e4
    datas = []

    quadwxp = QuadWxp()
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    # pre_qqqq = 0
    with open("train.txt", "w") as f:
        f.write('2\n')
        f.write('1\n') # xpp
        f.write('1\n') # q4u4
        for idx, data in enumerate(datas) :
            # wx = float(data[4])
            q0 = float(data[0])
            q1 = float(data[1])
            q2 = float(data[2])
            q3 = float(data[3])
            if idx == 0:
                pre_ww = q0*q2 + q1*q3
                continue
            rps_cw1 = GAIN*float(data[-4])
            rps_cw2 = GAIN*float(data[-3])
            rps_ccw1 = GAIN*float(data[-2])
            rps_ccw2 = GAIN*float(data[-1])
            qqqq = q0*q2 + q1*q3
            uuuu = rps_ccw1*abs(rps_ccw1) + rps_ccw2*abs(rps_ccw2) + rps_cw1*abs(rps_cw1) + rps_cw2*abs(rps_cw2) # 論文では下になってるのになぜかこれになってた
            q4u4 = qqqq*uuuu
            xpp = quadwxp.calc_f(q4u4)
            tmp = "\t".join([str(xpp), str(q4u4)])
            f.write(tmp+"\n")
            # pre_qqqq = q0*q2 + q1*q3
