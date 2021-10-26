import random
import math
from copy import deepcopy

class Quad() :
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
        self.RADIAN = 3.1415
        self.TWICE_RADIAN = 2*self.RADIAN

        self.u = [0 for _ in range(4)]
        self.var_p = [0 for _ in range(self.VAR_SIZE)]
        self.var_and_z_i = [0 for _ in range(self.VAR_SIZE)]
        self.var_and_z_i[0] = 1.0 # クオータニオンだから

        self.RBF_GAIN = 1/1024

    def calc(self, rps_cw1, rps_cw2, rps_ccw1, rps_ccw2) :
        self.u = [rps_cw1, rps_cw2, rps_ccw1, rps_ccw2]

        # 上から順にe0p, e1p, e2p, e3p
        self.var_p[0] = -0.5*self.var_and_z_i[1]*self.var_and_z_i[4] - 0.5*self.var_and_z_i[2]*self.var_and_z_i[5] - 0.5*self.var_and_z_i[3]*self.var_and_z_i[6]
        self.var_p[1] =  0.5*self.var_and_z_i[0]*self.var_and_z_i[4] + 0.5*self.var_and_z_i[2]*self.var_and_z_i[6] - 0.5*self.var_and_z_i[3]*self.var_and_z_i[5]
        self.var_p[2] =  0.5*self.var_and_z_i[0]*self.var_and_z_i[5] + 0.5*self.var_and_z_i[3]*self.var_and_z_i[4] - 0.5*self.var_and_z_i[1]*self.var_and_z_i[6]
        self.var_p[3] =  0.5*self.var_and_z_i[0]*self.var_and_z_i[6] + 0.5*self.var_and_z_i[1]*self.var_and_z_i[5] - 0.5*self.var_and_z_i[2]*self.var_and_z_i[4]

        # 上から順にwxp, wyp, wzp
        self.var_p[4] =  0.5*(2.0*(self.I_YY-self.I_ZZ)*self.var_and_z_i[5]*self.var_and_z_i[6]+self.ROTOR_DISTANCE*self.MAX_THRUST*(rps_ccw1*abs(rps_ccw1)-rps_ccw2*abs(rps_ccw2))/self.MAX_RPS_POW)/self.I_XX
        self.var_p[5] = -0.5*(2.0*(self.I_XX-self.I_ZZ)*self.var_and_z_i[4]*self.var_and_z_i[6]+self.ROTOR_DISTANCE*self.MAX_THRUST*(rps_cw1*abs(rps_cw1)-rps_cw2*abs(rps_cw2))/self.MAX_RPS_POW)/self.I_YY
        self.var_p[6] = ((self.I_XX-self.I_YY)*self.var_and_z_i[4]*self.var_and_z_i[5]-self.TORQUE_RATE*(rps_ccw1*abs(rps_ccw1)+rps_ccw2*abs(rps_ccw2)-rps_cw1*abs(rps_cw1)-rps_cw2*abs(rps_cw2)))/self.I_ZZ
        # print(self.var_p[4])
        # print(self.var_p[5])
        # print(self.var_p[6])

        # 上から順にxp, yp, zp
        self.var_p[7] = self.var_and_z_i[10]
        self.var_p[8] = self.var_and_z_i[11]
        self.var_p[9] = self.var_and_z_i[12]

        # 上から順にxpp, ypp, zpp
        self.var_p[10] =  2.0*self.MAX_THRUST*(self.var_and_z_i[0]*self.var_and_z_i[2]+self.var_and_z_i[1]*self.var_and_z_i[3])*(rps_ccw1*abs(rps_ccw1)+rps_ccw2*abs(rps_ccw2)+rps_cw1*abs(rps_cw1)+rps_cw2*abs(rps_cw2))/(self.MASS_OF_MACHINE*self.MAX_RPS_POW)
        self.var_p[11] = -2.0*self.MAX_THRUST*(self.var_and_z_i[0]*self.var_and_z_i[1]-self.var_and_z_i[2]*self.var_and_z_i[3])*(rps_ccw1*abs(rps_ccw1)+rps_ccw2*abs(rps_ccw2)+rps_cw1*abs(rps_cw1)+rps_cw2*abs(rps_cw2))/(self.MASS_OF_MACHINE*self.MAX_RPS_POW)
        self.var_p[12] = self.MAX_THRUST*(-1.0+2.0*self.var_and_z_i[0]*self.var_and_z_i[0]+2.0*self.var_and_z_i[3]*self.var_and_z_i[3])*(rps_ccw1*abs(rps_ccw1)+rps_ccw2*abs(rps_ccw2)+rps_cw1*abs(rps_cw1)+rps_cw2*abs(rps_cw2))/(self.MASS_OF_MACHINE*self.MAX_RPS_POW) - self.A_OF_GRAVITY

    def update(self) :
        def check_radian(rad) :
            if rad > self.RADIAN :
                return rad - self.TWICE_RADIAN
            elif rad < -self.RADIAN :
                return rad + self.TWICE_RADIAN
            else :
                return rad

        for i in range(self.VAR_SIZE) :
            self.var_and_z_i[i] += self.var_p[i]*self.dt

        # クオータニオン正規化
        # print(self.var_and_z_i[0:4])
        q_norm = math.sqrt( self.var_and_z_i[0]*self.var_and_z_i[0]+self.var_and_z_i[1]*self.var_and_z_i[1]+self.var_and_z_i[2]*self.var_and_z_i[2] + self.var_and_z_i[3]*self.var_and_z_i[3])
        # print(q_norm)
        self.var_and_z_i[0] /= q_norm
        self.var_and_z_i[1] /= q_norm
        self.var_and_z_i[2] /= q_norm
        self.var_and_z_i[3] /= q_norm
        self.var_and_z_i[4] = check_radian(self.var_and_z_i[4])
        self.var_and_z_i[5] = check_radian(self.var_and_z_i[5])
        self.var_and_z_i[6] = check_radian(self.var_and_z_i[6])
        # print(self.var_and_z_i[4:7])
        # for tmp in self.var_and_z_i[4:7] :
        #     if tmp > self.RADIAN or tmp < -self.RADIAN :
        #         print('radian error')

    def get_q(self) :
        obs = deepcopy(self.var_and_z_i[:4])
        for i in range(4) :
            obs[i] += random.gauss(0, 0.1)
        return obs

    def get_w(self) :
        obs = deepcopy(self.var_p[4:7])
        # for i in range(3) :
        #     obs[i] += random.gauss(0, 0.1)
        return obs

    def get_v(self) :
        obs = deepcopy(self.var_p[7:10])
        # for i in range(3) :
        #     obs[i] += random.gauss(0, 0.1)
        return obs

    def get_u(self) :
        obs = deepcopy(self.u)
        for i in range(4) :
            obs[i] *= self.RBF_GAIN
        return obs

def gen_file(vfile, wfile) :
    quad = Quad()
    data_len = 100000
    u_data = []
    v_data = []
    w_data = []
    for i in range(data_len) :
        # if i == 10 : exit()
        x = [random.uniform(0, 100) for _ in range(4)]
        quad.calc(*x)
        quad.update()
        # print(quad.get_w())
        u_data.append(quad.get_u())
        v_data.append(quad.get_v())
        w_data.append(quad.get_w())

    with open(vfile, 'w') as f:
        f.write('2\n3\n4\n')
        for v, u in zip(v_data, u_data) :
            f.write('\t'.join(list(map(str, v+u))) + '\n')

    with open(wfile, 'w') as f:
        f.write('2\n3\n4\n')
        for w, u in zip(w_data, u_data) :
            f.write('\t'.join(list(map(str, w+u))) + '\n')

def main() :
    gen_file('train_v.txt', 'train_w.txt')
    gen_file('test_v.txt', 'test_w.txt')

if __name__ == "__main__" :
    main()