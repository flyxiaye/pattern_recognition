#贝叶斯分类
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import math
import random
import draw

def fa(x, mu_f, mu_m, sigma_f, sigma_m):
    """bayes判别准则"""
    return 0.5 * (math.log(lg.det(sigma_m)) - math.log(lg.det(sigma_f)) - 
    (x - mu_f).reshape(1, -1) @ lg.inv(sigma_f) @ (x - mu_f).reshape(-1, 1) +
    (x - mu_m).reshape(1, -1) @ lg.inv(sigma_m) @ (x - mu_m).reshape(-1, 1))

def random_int_list(n, lo, hi):
    a = [0 for i in range(n)]
    for i in range(n):
        a[i] = random.randint(lo, hi)
    return a

def divide_data(data, idx, flag=1):
    """将输入的数据集选取特征并分成男女，返回处理后的数据集"""
    #flag=1,选取男女各选10个
    length = len(idx)
    am = [[] for i in range(length)]
    af = [[] for i in range(length)]
    for row in data:
        if row[-1] == 'm' or row[-1] == 'M':
            for i in range(length):
                am[i].append(float(row[idx[i]]))
        else:
            for i in range(length):
                af[i].append(float(row[idx[i]]))
    a_f = np.array(af)
    a_m = np.array(am)
    if flag:
        return a_f, a_m
    else:
        # naf, nam = np.zeros((length, 10)), np.zeros((length, 10))
        # rand_naf = random_int_list(10, 0, a_f.shape[1] - 1)
        # rand_nam = random_int_list(10, 0, a_m.shape[1] - 1)
        # for i in range(10):
        #     for j in range(length):
        #         naf[j, i] = a_f[j, rand_naf[i]]
        #         nam[j, i] = a_m[j, rand_nam[i]]
        # return naf, nam
        return a_f[:, 0:10], a_m[:, 0:10]


def validate_data(data, mu_f, mu_m, sigma_f, sigma_m, compartor):
    """将输入的数据集做验证，返回正确的样本数"""
    # compartor为输入的数据为哪个集合
    rights = 0
    nums = data.shape[1]
    dims = data.shape[0]
    for i in range(nums):
        x = data[:, i]
        f = fa(x, mu_f, mu_m, sigma_f, sigma_m)
        if f > 0 and compartor == 'female':
            rights += 1
        elif f < 0 and compartor == 'male':
            rights += 1  
    return rights


def read_data(filename):
    with open(filename) as f:
        data = []
        line = f.readline()
        while line:
            data.append(line.split())
            line = f.readline()
    return data

def computer_para(mu_f, mu_m, sigma_f, sigma_m):
    """计算决策面参数"""
    a = [0 for i in range(6)]
    sigma_f_1 = lg.inv(sigma_f)
    sigma_m_1 = lg.inv(sigma_m)
    W = -0.5 * sigma_f_1 + 0.5 * sigma_m_1
    w = sigma_f_1 @ mu_f.reshape(-1, 1) - sigma_m_1 @ mu_m.reshape(-1, 1)
    w0 = -0.5 * mu_f.reshape(1, -1) @ sigma_f_1 @ mu_f.reshape(-1, 1) \
        + 0.5 * mu_m.reshape(1, -1) @ sigma_m_1 @ mu_m.reshape(-1, 1) \
        - 0.5 * math.log(lg.det(sigma_f)) + 0.5 * math.log(lg.det(sigma_m))
    a[0] = W[0,0]
    a[1] = W[0,1] + W[1,0]
    a[2] = W[1,1]
    a[3] = w[0,0]
    a[4] = w[1,0]
    a[5] = -w0[0,0]
    return a
    

if __name__ == "__main__":
    idx = [i for i in range(2)]     #特征选取
    all_flag = 1                    #选择全部训练样本
    filename = 'dataset3.txt' #训练数据集
    vali_filename = 'vali_500_with_tag.txt' #验证数据集

    data = read_data(filename)
    vali_data = read_data(vali_filename)
    xfs, xms = divide_data(data, idx=idx, flag=all_flag)
    mu_f = np.mean(xfs, axis=1)
    mu_m = np.mean(xms, axis=1)
    cov_f = np.cov(xfs)
    cov_m = np.cov(xms)
    xfs, xms = divide_data(data=vali_data, idx=idx)
    rights_f = validate_data(xfs, mu_f, mu_m, cov_f, cov_m, 'female')
    rights_m = validate_data(xms, mu_f, mu_m, cov_f, cov_m, 'male')
    error = 1 - (rights_f + rights_m) / (xfs.shape[1] + xms.shape[1])
    print("bayes测试错误率：" + str(format(error, '.4f')))

    draw.draw_sample(xfs, xms, idx)
    a = computer_para(mu_f, mu_m, cov_f, cov_m)
    x_min = min(min(xfs[idx[0]]), min(xms[idx[0]]))
    x_max = max(max(xfs[idx[0]]), max(xms[idx[0]]))
    y_min = min(min(xfs[idx[1]]), min(xms[idx[1]]))
    y_max = max(max(xfs[idx[1]]), max(xms[idx[1]]))
    draw.draw_quadratic_curve(a, [x_min-10, x_max+10], [y_min-10, y_max+10])
    plt.show()



