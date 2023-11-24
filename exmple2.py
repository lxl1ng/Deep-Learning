import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import gen_batches

np.random.seed(2022)

train_image_file = 'data/MNIST/raw/train-images-idx3-ubyte'  # trainX
train_label_file = 'data/MNIST/raw/train-labels-idx1-ubyte'  # trainY，labels都是0到9的整数
test_image_file = 'data/MNIST/raw/t10k-images-idx3-ubyte'  # textX
test_label_file = 'data/MNIST/raw/t10k-labels-idx1-ubyte'  # textY

number0 = 0
number1 = 0


def decode_image(path: str) -> np.ndarray:  # decode_image函数的作用就是使一行有784个数据点
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, 784)  # 将28X28的图像变成784维的，因为是训练两个，那就是两行的784维
        images = np.array(images, dtype=float)  # 变成numpy数组的形式
    return images


def decode_label(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)  # 从文本或者二进制文件当中构建一个数组
        labels = np.array(labels, dtype=float)  # labels也组合成一个numpy
    return labels


def load_data() -> tuple:
    train_X = decode_image(train_image_file)  # trainX就是预测图像
    train_Y = decode_label(train_label_file)  # trainY就是预测标签
    test_X = decode_image(test_image_file)
    test_Y = decode_label(test_label_file)
    return (train_X, train_Y, test_X, test_Y)


trainX, trainY, testX, testY = load_data()

# digit1 and digit2 are two digits used for binary classification. You could change their values.
digit1 = 1  # 此设定的是手写数据，这是1的手写数据
digit2 = 7  # 此设定的也是手写数据，这是7的手写数据

idx = (trainY == digit1) + (trainY == digit2)  # 只是进行布尔运算，若是训练数据所得与手写相符，就是1，否则为0，所以idx最大为2，最小为0
trainX, trainY = trainX[idx, :], trainY[idx]
# 若idx判断出来为0，那么[idx,:]就代表是取得所有列的第1个元素，
# 若idx=1，则就代表取所有列的第2个元素，idx=2，就代表取所有列的第3个元素
# 经过实践验证，X[idx]默认取的是数组的第idx行的所有元素，也就是所有列的第idx+1的元素
num_train, num_feature = trainX.shape
# 此时的.shape是获取行列的维数，num_train就是行的维数，num_feature就是获取的列的维数，训练与特征的维度就是训练图像X的行列维度，num_train就代表是训练次数，num_feature是数字的特征
idx = (testY == digit1) + (testY == digit2)  # 同上，现在是进行测试了，上述都是训练数据
testX, testY = testX[idx, :], testY[idx]  # 记住X是图像而Y是标签，也就是说是给出答案的那个，label就是给出的答案项
num_test = testX.shape[0]  # num_test就=测试数据的行数

plt.figure(1, figsize=(20, 5))
for i in range(4):
    idx = np.random.choice(range(num_train))
    plt.subplot(int('14' + str(i + 1)))
    plt.imshow(trainX[idx, :].reshape((28, 28)))
    plt.title('label is %d' % trainY[idx])
# plt.show()

print('number of features is %d' % num_feature)
print('number of training samples is %d' % num_train)
print('number of testing samples is %d' % num_test)
trainY[np.where(trainY == digit1)], trainY[np.where(trainY == digit2)] = 0, 1
testY[np.where(testY == digit1)], testY[np.where(testY == digit2)] = 0, 1


class LogisticRegression():
    def __init__(self, num_feature: int, learning_rate: float) -> None:
        '''
        Constructor
        Parameters:
          num_features is the number of features.
          learning_rate is the learning rate.
        Return:
          there is no return value.
        '''
        self.num_feature = num_feature
        self.w = np.random.randn(num_feature + 1)
        self.learning_rate = learning_rate

    def artificial_feature(self, x: np.ndarray) -> np.ndarray:
        '''
        add one artificial features to the data input
        Parameters:
          x is the data input. x is one dimensional or two dimensional numpy array.
        Return:
          updated data input with the last column of x being 1s.
        '''
        if len(x.shape) == 1:  # if x is one dimensional, convert it to be two dimensional
            x = x.reshape((1, -1))
        #### write your code below ####
        # line_number = x.shape[1]
        one = np.ones((x.shape[0], 1))
        # X = np.insert(x, line_number, values = one, axis = 1)
        X = np.hstack((x, one))
        # 以上为两种加特征方式
        #### write yoru codel
        return X

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        '''
        Compute sigmoid activation function value by f(x*w)
        Parameters:
          x is data input with artificial features. x is a two dimensional numpy array.
        Return:
          one dimensional numpy array
        '''
        ### write your code below ###
        # first compute inner product between x and self.w
        # sencond, compute logistic function value of x*self.w
        # inx = np.dot(x, self.w)
        # if inx.any() >=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        #  prob = 1.0/(1 + np.exp(-inx))
        # else:
        #  prob =  exp(inx)/(1 + np.exp(inx))
        # ------------------------
        # inner = np.dot(x, self.w)
        # inner = np.clip(inner, -10000, None)
        # prob = 1 / (1 + np.exp(-inner))
        # ------------------------
        inner = np.dot(x, self.w)
        inner_ravel = inner.ravel()  # 将numpy数组展平
        length = len(inner_ravel)
        middle_number = []
        for index in range(length):
            if inner_ravel[index] >= 0:
                middle_number.append(1.0 / (1 + np.exp(-inner_ravel[index])))
            else:
                middle_number.append(np.exp(inner_ravel[index]) / (np.exp(inner_ravel[index]) + 1))
        prob = np.array(middle_number).reshape(inner.shape)
        # 以上防止exp爆炸
        # 对sigmoid函数的优化，避免了出现极大的数据溢出
        # 上方分割线两个算法实测会出现exp boom

        ### write your code above ###
        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict label probability for the input X
        Parameters:
          X is the data input. X is one dimensional or two dimensional numpy array.
        Return:
          predicted label probability, which is a one dimensional numpy array.
        '''
        X = self.artificial_feature(X)
        #### write your code below ####
        prob = self.sigmoid(X)  # 直接代入计算

        #### write your code above ####
        return prob

    def loss(self, y: np.ndarray, prob: np.ndarray) -> float:
        '''
        Compute cross entropy loss.
        Parameters:
          y is the true label. y is a one dimensional array.
          prob is the predicted label probability. prob is a one dimensional array.
        Return:
          cross entropy loss
        '''
        #### write your code below ####
        #### you must think of how to deal with the case that prob contains 1 or 0 ####

        # N = prob.shape[0] #若求平均则加上
        prob = np.clip(prob, 0.0001, 0.9999)  # 防止出现0,1的概率，出现之前即截断
        loss_value = (-1) * np.sum(y * np.log(prob) + (1 - y) * (np.log(1 - prob)))  # 公式
        # loss_value = np.sum(loss_value)

        #### write your code above ####
        return loss_value

    def gradient(self, trainX: np.ndarray, trainY: np.ndarray) -> np.ndarray:
        '''
        Compute gradient of logistic regression.
        Parameters:
          trainX is the training data input. trainX is a two two dimensional numpy array.
          trainY is the training data label. trainY is a one dimensional numpy array.
        Return:
          a one dimensional numpy array representing the gradient
        '''
        x = self.artificial_feature(trainX)
        #### write your code below ####
        # train_num = len(trainY)
        prob = self.sigmoid(x)
        # g = (1 / train_num) * np.dot(x.T, (prob - trainY))
        g = np.dot(x.T, (prob - trainY))
        # 两种求梯度的公式，其中一个是求平均

        #### write your code above ####
        return g

    def update_weight(self, dLdw: np.ndarray) -> None:
        '''
        Update parameters of logistic regression using the given gradient.
        Parameters:
          dLdw is a one dimensional gradient.
        Return:
          there is no return value
        '''
        self.w += -self.learning_rate * dLdw
        return

    def one_epoch(self, X: np.ndarray, Y: np.ndarray, batch_size: int, train: bool = True) -> tuple:
        '''
        One epoch of either training or testing procedure.
        Parameters:
          X is the data input. X is a two dimensional numpy array.
          Y is the data label. Y is a one dimensional numpy array.
          batch_size is the number of samples in each batch.
          train is a boolean value indicating training or testing procedure.
        Returns:
          loss_value is the average loss function value.
          acc is the prediction accuracy.
        '''
        num_sample = X.shape[0]  # number of samples
        num_correct = 0  # number of corrected predicted samples
        num_batch = int(num_sample / batch_size) + 1  # number of batch
        batch_index = list(gen_batches(num_sample, num_batch))  # index for each batch
        loss_value = 0  # loss function value
        for i, index in enumerate(batch_index):  # the ith batch
            x, y = X[index, :], Y[index]  # get a batch of samples
            if train:
                dLdw = self.gradient(x, y)  # compute gradient
                self.update_weight(dLdw)  # update parameters of the model
            prob = self.predict(x)  # predict the label probability
            loss_value += self.loss(y, prob) * x.shape[0]  # loss function value for ith batch
            num_correct += self.accuracy(y, prob) * x.shape[0]
        loss_value = loss_value / num_sample  # average loss
        acc = num_correct / num_sample  # accuracy
        return loss_value, acc

    def accuracy(self, y: np.ndarray, prob: np.ndarray) -> float:
        '''
        compute accuracy
        Parameters:
          y is the true label. y is a one dimensional array.
          prob is the predicted label probability. prob is a one dimensional array.
        Return:
          acc is the accuracy value
        '''
        #### write your code below ####
        margin = 0.5
        # N = np.size(prob)
        # for i in range(N):
        #  if np.array(prob)[i] < margin:
        #    np.array(prob)[i] = 0
        #  else:
        #    np.array(prob)[i] = 1
        # count = 0
        # for a,b in zip(y, prob):
        #  if a == b:
        #    count += 1
        # acc = 1.0 * count / (N) #准确个数除以总数

        # ----------以上为循环方式，不采用
        prob_label = np.around(prob)  # around四舍五入
        N = y.shape[0]  # 获取样本的数目
        acc = np.sum(y == prob_label) / N  # 求准确率
        #### write your code above ####
        return acc


def train(model, trainX, trainY, epoches, batch_size):
    loss_value, acc = model.one_epoch(trainX, trainY, batch_size, train=False)
    print('Initialization: ', 'loss %.4f  ' % loss_value, 'accuracy %.2f' % acc)
    for epoch in range(epoches):
        loss_value, acc = model.one_epoch(trainX, trainY, batch_size)
        print('epoch: %d' % (epoch + 1), 'loss %.4f  ' % loss_value, 'accuracy %.2f' % acc)


model = LogisticRegression(num_feature, learning_rate=0.01)
train(model, trainX, trainY, epoches=10, batch_size=256)

test_loss, test_acc = model.one_epoch(testX, testY, batch_size=256, train=False)
print('testing accuracy is %.4f' % test_acc)
