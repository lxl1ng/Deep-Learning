import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

input_size = 784
hidden_size = 128
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001
# 导入MNIST数据集，将数据转换为tensor（？）格式
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

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
class NeuralNet(nn.Module):
    # 初始化网络结构
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层，线性（liner）关系
        self.relu = nn.LeakyReLU()  # 隐藏层，使用ReLU函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 输出层，线性（liner）关系

    # forward 参数传递函数，网络中数据的流动
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size,hidden_size,num_classes)
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