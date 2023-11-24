import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

input_size = 784
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 64
num_classes = 10
num_epochs = 10
batch_size = 50
learning_rate = 0.001
# 导入MNIST数据集，将数据转换为tensor（？）格式
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
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model (1 hidden layer)

# 定义简单的前馈神经网络
class NeuralNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, n_hidden_3, n_hidden_4, out_dim):
        super(NeuralNet, self).__init__()  # super()

        # Frist:ReLU
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, out_dim),
            nn.ReLU())

        # Second:LeakyRelU
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, n_hidden_1),
        #     nn.LeakyReLU())
        # self.layer2 = nn.Sequential(
        #     nn.Linear(n_hidden_1, n_hidden_2),
        #     nn.LeakyReLU())
        # self.layer3 = nn.Sequential(
        #     nn.Linear(n_hidden_2, n_hidden_3),
        #     nn.LeakyReLU())
        # self.layer4 = nn.Sequential(
        #     nn.Linear(n_hidden_3, n_hidden_4),
        #     nn.LeakyReLU())
        # self.layer5 = nn.Sequential(
        #     nn.Linear(n_hidden_4, out_dim),
        #     nn.LeakyReLU())

        # Third:Sigmoid
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, n_hidden_1),
        #     nn.Sigmoid())
        # self.layer2 = nn.Sequential(
        #     nn.Linear(n_hidden_1, n_hidden_2),
        #     nn.Sigmoid())
        # self.layer3 = nn.Sequential(
        #     nn.Linear(n_hidden_2, n_hidden_3),
        #     nn.Sigmoid())
        # self.layer4 = nn.Sequential(
        #     nn.Linear(n_hidden_3, n_hidden_4),
        #     nn.Sigmoid())
        # self.layer5 = nn.Sequential(
        #     nn.Linear(n_hidden_4, out_dim),
        #     nn.Sigmoid())

        # Forth:Tanh
        # self.layer1 = nn.Sequential(
        #     nn.Linear(in_dim, n_hidden_1),
        #     nn.Tanh())
        # self.layer2 = nn.Sequential(
        #     nn.Linear(n_hidden_1, n_hidden_2),
        #     nn.Tanh())
        # self.layer3 = nn.Sequential(
        #     nn.Linear(n_hidden_2, n_hidden_3),
        #     nn.Tanh())
        # self.layer4 = nn.Sequential(
        #     nn.Linear(n_hidden_3, n_hidden_4),
        #     nn.Tanh())
        # self.layer5 = nn.Sequential(
        #     nn.Linear(n_hidden_4, out_dim),
        #     nn.Tanh())

    # 定义向前传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


model = NeuralNet(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像数组重新调整形状的操作？
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

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

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # torch.max函数返回两个值（返回最大数的位置）
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('预测准确率 ：{} %'.format(100 * correct / total))

    x, y = data
    x, y = x.to(device), y.to(device)
    # 得到模型预测输出，10个输出，即该图片为每个数字的概率
    res = model(x)
    # 最大概率的就为预测值
    r = torch.argmax(res)
    l = y.item()
    sum += 1 if r == l else 0