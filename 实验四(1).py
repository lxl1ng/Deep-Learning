import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# 分批次训练，一批 100 个训练数据
input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 64
num_classes = 10
num_epochs = 2
batch_size = 50
learning_rate = 0.001
# 架加载数据集
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           )

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 分批
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)


# test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)


# 定义CNN神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1：输入通道为 1，输出通道为 6，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # 卷积层2：输入通道为 6，输出通道为 16，卷积核大小 为 5
        # 使用 Relu 激活函数
        # 使用最大值池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # 输出层，全连接层，输入大小 16 * 7 * 7， 输出大小 84
        self.layer_out1 = nn.Linear(16 * 7 * 7, 84)
        self.layer_out2 = nn.Linear(84 * 1 * 1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.layer_out1(x)
        self.out = self.layer_out2(x)
        return self.out


# 实例化CNN，并将模型放在 GPU 上训练
model = CNN().to(device)
summary(model, input_size=(1,28,28))
# 使用交叉熵损失，同样，将损失函数放在 GPU 上
loss_fn = nn.CrossEntropyLoss().to(device)
# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        # 调用模型预测
        output = model(x).to(device)
        # 计算损失值
        loss = loss_fn(output, y.long())

        # CSDN：
        # 作用：清除优化器关于所有参数x的累计梯度值 x∗grad，一般在loss.backward()前使用
        # 作用：将损失loss向输入测进行反向传播；
        # 作用：利用优化器对参数x进行更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

sum = 0
# test：
for i, data in enumerate(test_loader):
    images, labels = data
    x, y = images.to(device), labels.to(device)
    # 得到模型预测输出，10个输出，即该图片为每个数字的概率
    res = model(x)
    # 最大概率的就为预测值
    r = torch.argmax(res)
    l = y.item()
    sum += 1 if r == l else 0
    print(f'test({i})     CNN:{r} -- label:{l}')

print('accuracy：', sum / 10000)
