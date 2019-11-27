#Fisher线性分类器
import numpy as np
import matplotlib.pyplot as plt
import bayes
import draw


def inner_s(data, mu):
    """计算类内离散度矩阵"""
    dim = data.shape[0]
    num = data.shape[1]
    s = np.zeros((dim, dim))
    for i in range(num):
        x = data[:, i] - mu
        s += np.dot(x.reshape(-1, 1), x.reshape(1, -1))
    return s


def classify(data, w, w0, comparator):
    """测试集的验证"""
    rights = 0
    num = data.shape[1]
    dim = data.shape[0]
    for i in range(num):
        gx = np.dot(data[:, i].reshape(1,-1), w) + w0
        if gx > 0 and comparator == 'female':
            rights += 1
        elif gx < 0 and comparator == 'male':
            rights += 1
    return rights

def computer_para(w, w0):
    a = [0 for i in range(3)]
    a[0] = w[0,0]
    a[1] = w[1,0]
    a[2] = -w0[0,0]
    print(a)
    return a

if __name__ == "__main__": 
    idx = [i for i in range(2)]     #特征选取
    all_flag = 1                    #选择全部训练样本
    filename = 'dataset3.txt'
    vali_filename = 'vali_500_with_tag.txt'
    data = bayes.read_data(filename)
    vali_data = bayes.read_data(vali_filename)
    xfs, xms = bayes.divide_data(data, idx=idx, flag=all_flag)
    mu_female = np.mean(xfs, axis=1)
    mu_male = np.mean(xms, axis=1)
    s_female = inner_s(xfs, mu_female)
    s_male = inner_s(xms, mu_male)
    sw= s_female + s_male       #总类内离散度矩阵
    w = np.linalg.inv(sw) @ (mu_female - mu_male).reshape(-1,1)#权向量方向
    w0 = -0.5 * (mu_female + mu_male).reshape(1, -1) @ w        #阈值
    xfs, xms = bayes.divide_data(vali_data, idx=idx)
    right_f = classify(xfs, w, w0, 'female')
    right_m = classify(xms, w, w0, 'male')
    error = 1 - (right_f + right_m) / (xfs.shape[1] + xms.shape[1])
    print("Fisher测试错误率：" + str(format(error,'.4f')))

    a = computer_para(w, w0)
    draw.draw_sample(xfs, xms, idx)
    x_min = min(min(xfs[idx[0]]), min(xms[idx[0]]))
    x_max = max(max(xfs[idx[0]]), max(xms[idx[0]]))
    y_min = min(min(xfs[idx[1]]), min(xms[idx[1]]))
    y_max = max(max(xfs[idx[1]]), max(xms[idx[1]]))
    draw.draw_quadratic_curve(a, (x_min-10, x_max+10), (y_min-10, y_max+10))
    plt.show()