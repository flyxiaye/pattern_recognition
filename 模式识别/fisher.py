#Fisher线性分类器
import numpy as np
import matplotlib.pyplot as plt
import bayes
import draw
import mlp


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
    return a


def predict_data(w, w0, idx):
    """对无标签数据进行预测，并输出至文件夹中"""
    pred_filename = 'vali_100_no_tag.txt'   #预测数据集
    write_filename = 'pred_by_fisher.txt'
    prediction_data = bayes.read_data(pred_filename)
    pred_data, pred_labels = mlp.load_data(prediction_data, idx)
    i = 0
    while i < len(pred_labels):
        gx = np.dot(pred_data[i].reshape(1, -1), w) + w0
        if gx > 0:
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
    data = bayes.read_data(train_filename)
    vali_data = bayes.read_data(test_filename)
    xfs, xms = bayes.divide_data(data, idx=idx, flag=all_flag)
    mu_female = np.mean(xfs, axis=1)
    mu_male = np.mean(xms, axis=1)
    s_female = inner_s(xfs, mu_female)
    s_male = inner_s(xms, mu_male)
    sw= s_female + s_male       #总类内离散度矩阵
    w = np.linalg.inv(sw) @ (mu_female - mu_male).reshape(-1,1)#权向量方向
    w0 = -0.5 * (mu_female + mu_male).reshape(1, -1) @ w        #阈值
    xfs_val, xms_val = bayes.divide_data(vali_data, idx=idx)
    right_f = classify(xfs_val, w, w0, 'female')
    right_m = classify(xms_val, w, w0, 'male')
    error = 1 - (right_f + right_m) / (xfs_val.shape[1] + xms_val.shape[1])
    print('选取训练样本个数:' + str(xfs.shape[1] + xms.shape[1]))
    print('选取特征维数：' + str(idx))
    print("Fisher测试错误率：" + str(format(error,'.4f')))
    print()

    #绘制图形
    if len(idx) == 2:
        a = computer_para(w, w0)
        draw.draw_sample(xfs, xms, idx)
        x_min = min(min(xfs[0]), min(xms[0]))
        x_max = max(max(xfs[0]), max(xms[0]))
        y_min = min(min(xfs[1]), min(xms[1]))
        y_max = max(max(xfs[1]), max(xms[1]))
        draw.draw_curve(a, (x_min-10, x_max+10), (y_min-10, y_max+10))
        plt.title('size of samples:' + str(xms.shape[1] + xfs.shape[1]))
        plt.show()

    #预测数据 在全部样本数 全部特征情况下
    #if all_flag == 1 and len(idx) == 10:
    #    predict_data(w, w0, idx)


if __name__ == "__main__": 
    filename = 'dataset3.txt'
    vali_filename = 'vali_500_with_tag.txt'
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
    