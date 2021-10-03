'''
角速度をクオータニオン形式に変更して保存
'''
def euler2quat(x, y, z, q0, q1, q2, q3) :
    # 移動体座標系
    # https://www.kazetest.com/vcmemo/quaternion/quaternion.htm
    dq0 = -q1*x - q2*y - q3*z
    dq1 =  q0*x + q2*z - q3*y
    dq2 =  q0*y - q1*z + q3*x
    dq3 =  q0*z + q1*y - q2*x
    return [str(dq0/2), str(dq1/2), str(dq2/2), str(dq3/2)]

if __name__ == "__main__" :
    datas = []
    with open("quat_rate_sysin.txt", "r") as f:
        datas = [s.strip().split() for s in f.readlines()]

    with open("train.txt", "w") as f:
        for i, data in enumerate(datas) :
            tmp = 0
            if i == 0 :
                tmp = '2'
            elif i == 1 :
                tmp = '4' # クオータニオンなので4
            elif i == 2 :
                tmp = '4'
            else :
                quat = euler2quat(*list(map(float, data[4:7])), *list(map(float, data[0:4])))
                tmp = "\t".join(quat + data[-4:])
            f.write(tmp+"\n")