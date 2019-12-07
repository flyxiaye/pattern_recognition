#贝叶斯分类
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import math
import random
import draw
import mlp

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
    

def predict_data(mu_f, mu_m, cov_f, cov_m, idx):
    """对无标签数据进行预测，并输出至文件夹中"""
    pred_filename = 'vali_100_no_tag.txt'   #预测数据集
    write_filename = 'pred_by_bayes.txt'
    prediction_data = read_data(pred_filename)
    pred_data, pred_labels = mlp.load_data(prediction_data, idx)
    i = 0
    while i < len(pred_labels):
        f = fa(pred_data[i], mu_f, mu_m, cov_f, cov_m)
        if f > 0:
            prediction_data[i].append('f')
        else:
            prediction_data[i].append('m')
        i += 1
    with open(write_filename, 'w') as f:
        for row in prediction_data:
            for elem in row:
                f.write(str(elem) + '\t')
            f.write('\n')


def run(train_filename, test_filename, idx=[i for i in range(10)], all_flag=1):
    data = read_data(train_filename)
    vali_data = read_data(test_filename)
    xfs, xms = divide_data(data, idx=idx, flag=all_flag)
    mu_f = np.mean(xfs, axis=1)
    mu_m = np.mean(xms, axis=1)
    cov_f = np.cov(xfs)
    cov_m = np.cov(xms)
    xfs_val, xms_val = divide_data(data=vali_data, idx=idx)
    rights_f = validate_data(xfs_val, mu_f, mu_m, cov_f, cov_m, 'female')
    rights_m = validate_data(xms_val, mu_f, mu_m, cov_f, cov_m, 'male')
    error = 1 - (rights_f + rights_m) / (xfs_val.shape[1] + xms_val.shape[1])
    print('选取训练样本个数:' + str(xfs.shape[1] + xms.shape[1]))
    print('选取特征维数：' + str(idx))
    print("bayes测试错误率：" + str(format(error, '.4f')))
    print()

    if len(idx) == 2:
        draw.draw_sample(xfs, xms, idx)
        a = computer_para(mu_f, mu_m, cov_f, cov_m)
        x_min = min(min(xfs[0]), min(xms[0]))
        x_max = max(max(xfs[0]), max(xms[0]))
        y_min = min(min(xfs[1]), min(xms[1]))
        y_max = max(max(xfs[1]), max(xms[1]))
        draw.draw_curve(a, [x_min-10, x_max+10], [y_min-10, y_max+10])
        plt.title('size of samples:' + str(xms.shape[1] + xfs.shape[1]))
        plt.show()

    #预测数据 在全部样本数 全部特征情况下
    if all_flag == 1 and len(idx) == 10:
        predict_data(mu_f, mu_m, cov_f, cov_m, idx)


if __name__ == "__main__":
    filename = 'dataset3.txt' #训练数据集
    vali_filename = 'vali_500_with_tag.txt' #验证数据集
    #2特征全部样本
    idx = [i for i in range(2)]     #特征选取
    all_flag = 1                    #选择全部训练样本
    run(filename, vali_filename, idx, all_flag)
    #10特征全部样本
    idx = [i for i in range(10)]
    all_flag = 1
    run(filename, vali_filename, idx, all_flag)
    #2特征20个样本
    idx = [i for i in range(2)]
    all_flag = 0
    run(filename, vali_filename, idx, all_flag)
    #10特征20个样本
    idx = [i for i in range(10)]
    all_flag = 0
    run(filename, vali_filename, idx, all_flag)

   
