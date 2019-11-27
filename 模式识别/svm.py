#支持向量机
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
import bayes


def get_train_data(data, idx):
    """输入数据处理"""
    length = len(idx)
    num = len(data[0])
    x = []
    y = []
    for row in data:
        x.append([float(row[idx[i]]) for i in range(length)])
        if row[-1] == 'f' or row[-1] == 'F':
            y.append(1)
        else:
            y.append(0)
    return np.array(x), np.array(y)


if __name__ == "__main__":
    filename = 'dataset3.txt'
    vali_filename = 'vali_500_with_tag.txt'
    data = bayes.read_data(filename)
    vali_data = bayes.read_data(vali_filename)
    idx = [i for i in range(2)]
    x, y = get_train_data(data, idx)
    clf = svm.SVC(C=1.0, kernel='linear', gamma='scale')
    clf.fit(x, y)
    x, y = get_train_data(vali_data, idx)
    print('svm测试错误率：' + str(format(1 - clf.score(x, y), '.4f')))


    plt.scatter(x[:, 0], x[:, 1], c=y, s=3, cmap=plt.cm.Paired)
    plt.show()

