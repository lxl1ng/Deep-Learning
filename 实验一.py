import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

input_size = 28 * 28
num_classes = 10  # 分类的数量（0-9）共10种
num_epochs = 10  # 训练次数
batch_size = 10  # 每批容量多大
learning_rate = 0.05

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

# 利用torch.nn提供的逻辑回归模型
model = nn.Linear(input_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像数组重新调整形状的操作？
        images = images.reshape(-1, input_size)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

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

# 测试模型
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

