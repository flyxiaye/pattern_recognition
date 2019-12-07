#支持向量机
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
import bayes


def get_train_data(data, idx, flag=1):
    """输入数据处理"""
    length = len(idx)
    res_data = []
    res_label = []
    if flag == 1:     
        for row in data:
            res_data.append([float(row[idx[i]]) for i in range(length)])
            if row[-1] == 'm' or row[-1] == 'M':
                res_label.append(1)
            else:
                res_label.append(0)
    else:
        male_num, female_num = 0, 0
        for row in data:
            if male_num >= 10 and female_num >= 10:
                break
            if (row[-1] == 'm' or row[-1] == 'M') and male_num < 10:
                res_label.append(1)
                res_data.append([float(row[idx[i]]) for i in range(length)])
                male_num += 1
            elif female_num < 10:
                res_label.append(0)
                res_data.append([float(row[idx[i]]) for i in range(length)])
                female_num += 1
    return np.array(res_data), np.array(res_label)


def divide_dataset(data, labels, split=3):
    """交叉验证划分数据集"""
    length = data.shape[0]
    idx = np.random.permutation(length)
    dataset = [[] for i in range(split)]
    labelsset = [[] for i in range(split)]
    for i in range(split):
        dataset[i] = data[idx[int(length * i / split): int(length * (i + 1) / split)]]
        labelsset[i] = labels[idx[int(length * i / split): int(length * (i + 1) / split)]]
    return dataset, labelsset


def cross_validation(clf, data, labels, dim, fold=3, epoch=10):
    """
        交叉验证
        epoch：交叉验证的次数
        fold： 交叉验证的折数
    """
    accuracies = 0;
    for i in range(epoch):
        data_set, labels_set = divide_dataset(data[:], labels[:], split=fold)
        accuracy = 0
        for i in range(fold):
            val_data, val_labels = data_set[i], labels_set[i]
            train_data, train_labels = np.empty(shape=[0, data_set[i].shape[1]]), []
            for j in range(fold):
                if j != i:
                    train_data = np.concatenate((train_data, data_set[j]), axis=0)
                    train_labels = np.concatenate((train_labels, labels_set[j]))
            clf.fit(train_data, train_labels)
            accuracy += clf.score(val_data, val_labels)
        accuracies += accuracy / fold
    return accuracies / epoch
        

def border_of_classifier(sklearn_cl, x, y):
    """
    param sklearn_cl: sklearn的分类器
    param x: np.array
    param y: np.array
    """
    x_min, y_min = x.min(axis=0) - 1
    x_max, y_max = x.max(axis=0) + 1
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                    np.arange(y_min, y_max, 0.1))
    mesh_output = sklearn_cl.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    plt.pcolormesh(x_values, y_values, mesh_output, cmap = 'rainbow')
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, edgecolors='steelblue', linewidth=1, cmap=plt.cm.Spectral)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.title('size of sample:' + str(y.shape[0]), fontsize=24)


def predict_data(model, idx):
    """对无标签数据进行预测，并输出至文件夹中"""
    pred_filename = 'vali_100_no_tag.txt'   #预测数据集
    write_filename = 'pred_by_svm.txt'
    prediction_data = bayes.read_data(pred_filename)
    pred_data, pred_labels = get_train_data(prediction_data, idx)
    pred_labels = model.predict(pred_data)
    i = 0
    while i < len(pred_labels):
        if 1 == pred_labels[i]:
            prediction_data[i].append('m')
        else:
            prediction_data[i].append('f')
        i += 1
    with open(write_filename, 'w') as f:
        for row in prediction_data:
            for elem in row:
                f.write(str(elem) + '\t')
            f.write('\n')


def run(train_filename, test_filename, idx=[i for i in range(10)], all_flag=1):
    data = bayes.read_data(train_filename)
    vali_data = bayes.read_data(test_filename)
    train_data, train_labels = get_train_data(data, idx, all_flag)
    
    #交叉验证
    #paras = np.arange(start=0.0001, stop=0.001, step=0.0001)
    #accuracies = np.empty(shape=[0, 2])
    #for para in paras:
    #    clf = svm.SVC(C=2.0, kernel='rbf', gamma=para, degree=3)
    #    accuracy = cross_validation(clf, train_data, train_labels, len(idx))
    #    accuracies = np.concatenate((accuracies, [[para, accuracy]]))
    #print(accuracies)
    #plt.plot(accuracies[:,0], accuracies[:,1])
    #plt.show()

    clf = svm.SVC(C=2.0, kernel='rbf', gamma=0.0007, degree=2)
    clf.fit(train_data, train_labels)
    test_data, test_labels = get_train_data(vali_data, idx)
    print('选取训练样本个数：' + str(train_labels.shape[0]))
    print('选取特征：' + str(idx))
    print('svm测试错误率：' + str(format(1 - clf.score(test_data, test_labels), '.4f')))
    print()
    
    #绘制分类边界
    if len(idx) == 2:
        border_of_classifier(clf, train_data, train_labels)
        plt.show()

    #预测数据 在全部样本数 全部特征情况下
    #if all_flag == 1 and len(idx) == 10:
    #    predict_data(clf, idx)

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
    

