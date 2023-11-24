import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.datasets import IMDB

# 定义字段（Field）
TEXT = data.Field(tokenize='spacy', include_lengths=True, tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载IMDB数据集
train_data, test_data = IMDB.splits(TEXT, LABEL)

# 创建训练集和验证集
train_data, valid_data = train_data.split(split_ratio=0.8)  # 80% 训练，20% 验证

# 构建词汇表
MAX_VOCAB_SIZE = 25000
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# 定义迭代器
BATCH_SIZE = 32
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device
)


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input, embedding, hidden, output):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input, embedding)
        self.rnn = nn.RNN(embedding, hidden)
        self.fc = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        predictions = self.relu(self.fc(hidden.squeeze(0)))
        predictions = self.fc2(predictions)
        return predictions


# 初始化模型和优化器
INPUT = len(TEXT.vocab)
EMBEDDING = 100
HIDDEN = 128
OUTPUT = 1
model = RNN(INPUT, EMBEDDING, HIDDEN, OUTPUT)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 将模型和损失函数转移到GPU
model = model.to(device)
criterion = criterion.to(device)


# 计算准确率
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练函数
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text[0]).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 验证函数
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text[0]).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# 训练模型
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    print(f'Epoch:{epoch + 1:02}')
    print(f'\tTrain Loss:{train_loss:.3f}|Train Acc:{train_acc * 100:.2f}%')
    print(f'\tValidation Loss:{valid_loss:.3f}|Validation Acc:{valid_acc * 100:.2f}%')

# 测试模型
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss:{test_loss:.3f}|Test Acc:{test_acc * 100:.2f}%')

# 使用模型进行预测
nlp = spacy.load('en_core_web_sm')


def predict_sentiment(model, sentence, threshold=0.5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    if prediction.item() > threshold:
        return "正面评论"
    else:
        return "负面评论"


# 示例预测
positive_review = "I loved this anime, it's interesting"
negative_review = "This anime is fuking shit"
pp = "I loved this movie!It was amazing"
pw = "This book is boring.I hate it."
print(f'Positive review prediction: {predict_sentiment(model, positive_review)}')
print(f'Negative review prediction: {predict_sentiment(model, negative_review)}')
print(f'Positive review prediction: {predict_sentiment(model, pp)}')
print(f'Negative review prediction: {predict_sentiment(model, pw)}')
