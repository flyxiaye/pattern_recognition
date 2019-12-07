import tensorflow as tf
import numpy as np
from tensorflow import keras
import bayes
import datetime
import matplotlib.pyplot as plt


def load_data(data, idx, flag=1):
    """将输入的数据集选取特征并分成男女，返回处理后的数据集"""
    #flag=1,选取男女各选10个
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


def data_normalization(data, max, min):
    length = len(max)
    for i in range(length):
        data[:,i]  = (data[:, i] - min[i]) / (max[i] - min[i])


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

def create_model(dim, learning_rate=0.1, units=16):
    model = keras.Sequential([
        keras.layers.Dense(units, input_shape=(dim,), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
        ])
    model.compile(loss=keras.losses.MeanSquaredError(),
                optimizer=keras.optimizers.SGD(learning_rate),
                metrics=['accuracy']) # 生成模型
    return model


def cross_validation(data, labels, dim, fold=3):
    data, labels = divide_dataset(data, labels, split=fold)
    rates = np.arange(start=0.05, stop=5, step=0.05)
    accuracies = np.empty(shape=[0,2])
    for rate in rates:
        accuracy = 0.0
        for i in range(fold):
            val_data, val_labels = data[i], labels[i]
            train_data, train_labels = np.empty(shape=[0, data[i].shape[1]]), []
            for j in range(fold):
                if j != i:
                    train_data = np.concatenate((train_data, data[j]), axis=0)
                    train_labels = np.concatenate((train_labels, labels[j]))
            model = create_model(dim, learning_rate=rate)
            model.summary()
            history = model.fit(train_data, train_labels, batch_size = 16, epochs=50, verbose=0)
            results = model.evaluate(val_data, val_labels, verbose=0)
            accuracy += results[1]
        accuracy /= fold
        accuracies = np.concatenate((accuracies, [[rate, accuracy]]))
    return accuracies


def predict_data(model, idx, max, min):
    """对无标签数据进行预测，并输出至文件夹中"""
    pred_filename = 'vali_100_no_tag.txt'   #预测数据集
    write_filename = 'pred_by_mlp.txt'
    prediction_data = bayes.read_data(pred_filename)
    pred_data, pred_labels = load_data(prediction_data, idx)
    data_normalization(pred_data, max, min)
    predictions = model.predict(pred_data)
    i = 0
    while i < len(predictions):
        if predictions[i] > 0.5:
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
    #读取数据
    train_data = bayes.read_data(train_filename)
    test_data = bayes.read_data(test_filename)

    #数据分割处理
    train_data, train_labels = load_data(train_data, idx, all_flag)
    test_data, test_labels = load_data(test_data, idx)

    #数据归一化
    dim = len(idx)  #特征总维度数
    max_d = [max(max(train_data[:,i]), max(test_data[:,i])) for i in range(dim)]
    min_d = [min(min(train_data[:,i]), min(test_data[:,i])) for i in range(dim)]
    data_normalization(train_data, max_d, min_d)
    data_normalization(test_data, max_d, min_d)

    #交叉验证
    #accuracies = cross_validation(train_data, train_labels, dim)
    #plt.plot(accuracies[:,0], accuracies[:,1])
    #plt.show()

    #测试数据
    model = create_model(dim)
    tensorboard_callback = tf.keras.callbacks.TensorBoard()
    history = model.fit(train_data, train_labels, batch_size = 16, epochs=100, verbose=0, callbacks=[tensorboard_callback])
    results = model.evaluate(test_data, test_labels, verbose=0)

    print('选取训练样本个数：' + str(train_labels.shape[0]))
    print('选取特征：' + str(idx))
    print('MLP测试错误率：' + str(format((1 - results[1]), '.4f')))
    print()
    #print(results)

    #预测数据 在全部样本数 全部特征情况下
    #if all_flag == 1 and len(idx) == 10:
    #    predict_data(model, idx, max_d, min_d)
    

if __name__ == '__main__':
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