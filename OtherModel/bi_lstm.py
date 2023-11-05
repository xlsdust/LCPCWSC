import torch
import torch.nn as nn


# 定义双向LSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM输出的hidden_size乘以2

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 双向LSTM的隐藏状态乘以2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出作为双向LSTM的输出
        out = self.fc(out[:, -1, :])

        return out

if __name__ == "__main__":
    # 定义输入
    batch_size = 10
    seq_len = 20
    hidden_size = 64
    input_size = 32
    num_layers = 2
    num_classes = 2

    # 创建双向LSTM模型实例
    model = BiLSTM(input_size, hidden_size, num_layers, num_classes)

    # 生成随机输入数据
    input_data = torch.randn(batch_size, seq_len, input_size)

    # 前向传播
    output = model(input_data)
    print(output.shape)  # 输出为(batch_size, num_classes)
