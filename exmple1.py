# -*- coding = utf-8 -*-
"""
#@Time:2022年4月1日
#@Author:YuDai
#@File:逻辑回归算法minst.py
#@Software : pycharm
"""

import struct  # 这是一个如何定义格式字符串的库
import time

import numpy as np
from numpy import *
from scipy.special import expit  # logistic sigmoid函数 是一个逻辑回归算法里需要用到的函数（贼复杂，看都看不懂）


# 读取图片
def read_image(file_name):
    # 先用二进制方式把文件都读进来
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    offset = 0
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽 取前4个整数，返回一个元组
    head = struct.unpack_from('>IIII', file_content, offset)
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  # 图片数
    rows = head[2]  # 宽度
    cols = head[3]  # 高度

    images = np.empty((imgNum, 784))  # empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size = rows * cols  # 单个图片的大小

    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。
    # 这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）

    fmt = '>' + str(image_size) + 'B'  # 单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        offset += struct.calcsize(fmt)
    return images


# 读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    # 和上面一样解析数据

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)



def loadDataSet():
    train_x_filename = r"data/MNIST/raw/train-images-idx3-ubyte"
    train_y_filename = r"data/MNIST/raw/train-labels-idx1-ubyte"
    test_x_filename = r"data/MNIST/raw/t10k-images-idx3-ubyte"
    test_y_filename = r"data/MNIST/raw/t10k-labels-idx1-ubyte"
    train_x = read_image(train_x_filename)
    train_y = read_label(train_y_filename)
    test_x = read_image(test_x_filename)
    test_y = read_label(test_y_filename)

    # # # #调试的时候让速度快点，就先减少数据集大小
    # train_x=train_x[0:1000,:]
    # train_y=train_y[0:1000]
    # test_x=test_x[0:500,:]
    # test_y=test_y[0:500]

    return train_x, test_x, train_y, test_y


# 从这开始就是有关sigmoid函数的运算
# https://blog.csdn.net/qq_39783601/article/details/105557388?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164924372916780366571823%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164924372916780366571823&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-105557388.142^v5^pc_search_result_cache,157^v4^control&utm_term=sigmoid&spm=1018.2226.3001.4187
# Sigmoid函数的图像一般来说并不直观，我理解的是对数值越大，函数越逼近1，数值越小，函数越逼近0，将数值结果转化为了0到1之间的概率
# sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 预测函数
def classifyVector(inX, weights):  # 这里的inX相当于test_data,以回归系数和特征向量作为输入来计算对应的sigmoid
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 训练模型
def train_model(train_x, train_y, theta, learning_rate, iterationNum, numClass):  # theta是n+1行的列向量
    m = train_x.shape[0]
    n = train_x.shape[1]
    train_x = np.insert(train_x, 0, values=1, axis=1)
    J_theta = np.zeros((iterationNum, numClass))

    for k in range(numClass):
        # print(k)
        real_y = np.zeros((m, 1))
        index = train_y == k  # index中存放的是train_y中等于0的索引
        real_y[index] = 1  # 在real_y中修改相应的index对应的值为1，先分类0和非0

        for j in range(iterationNum):
            # print(j)
            temp_theta = theta[:, k].reshape((785, 1))
            # h_theta=expit(np.dot(train_x,theta[:,k]))#是m*1的矩阵（列向量）,这是概率
            h_theta = expit(np.dot(train_x, temp_theta)).reshape((60000, 1))
            # 这里的一个问题，将train_y变成0或者1
            # J_theta[j, k] = (np.dot(np.log(h_theta).T, real_y) + np.dot((1 - real_y).T, np.log(1 - h_theta))) / (-m)
            temp_theta = temp_theta + learning_rate * np.dot(train_x.T, (real_y - h_theta))

            # theta[:,k] =learning_rate*np.dot(train_x.T,(real_y-h_theta))
            theta[:, k] = temp_theta.reshape((785,))

    return theta  # 返回的theta是n*numClass矩阵


def predict(test_x, test_y, theta, numClass):  # 这里的theta是学习得来的最好的theta，是n*numClass的矩阵
    errorCount = 0
    test_x = np.insert(test_x, 0, values=1, axis=1)
    m = test_x.shape[0]

    h_theta = expit(np.dot(test_x, theta))  # h_theta是m*numClass的矩阵，因为test_x是m*n，theta是n*numClass
    h_theta_max = h_theta.max(axis=1)  # 获得每行的最大值,h_theta_max是m*1的矩阵，列向量
    h_theta_max_postion = h_theta.argmax(axis=1)  # 获得每行的最大值的label
    for i in range(m):
        if test_y[i] != h_theta_max_postion[i]:
            errorCount += 1

    error_rate = float(errorCount) / m
    print("error_rate", error_rate)
    return error_rate


def mulitPredict(test_x, test_y, theta, iteration):
    numPredict = 10
    errorSum = 0
    for k in range(numPredict):
        errorSum += predict(test_x, test_y, theta, iteration)
    print("after %d iterations the average error rate is:%f" % (numPredict, errorSum / float(numPredict)))


if __name__ == '__main__':
    print("开始读入训练数据。。。")
    time1 = time.time()
    train_x, test_x, train_y, test_y = loadDataSet()
    time2 = time.time()
    print("读入时间cost：", time2 - time1, "second")

    numClass = 10  # 控制列向量
    iteration = 1000  # 这里可以改迭代的次数，不同的迭代次数错误率也不一样，迭代次数越多，error 就越小

    learning_rate = 0.001  # 学习率
    n = test_x.shape[1] + 1

    theta = np.zeros(
        (n, numClass))  # theta=np.random.rand(n,1)#随机构造n*numClass的矩阵,因为有numClass个分类器，所以应该返回的是numClass个列向量（n*1）

    print("开始训练数据。。。")
    theta_new = train_model(train_x, train_y, theta, learning_rate, iteration, numClass)
    time3 = time.time()
    print("训练时间cost：", time3 - time2, "second")

    print("开始预测数据。。。。")
    predict(test_x, test_y, theta_new, iteration)
    time4 = time.time()
    print("预测时间cost", time4 - time3, "second")
