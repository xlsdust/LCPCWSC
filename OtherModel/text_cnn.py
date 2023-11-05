import torch
import torch.nn as nn
from mycode.model.OtherModel import *
class TextCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)

    def forward(self, x,mask=None):
        # x: [batch_size, seq_len, hidden_size]
        # mask: [batch_size,seq_len]  只有文本部分为1，mask部分为0，如果为None，代表不使用mean pooling
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]

        x = [conv(x) for conv in self.convs]  # Apply convolutional filters

        x_0 = [torch.relu(conv_out).max(dim=2)[0] for conv_out in x]  # Max pooling
        x_1 = [torch.relu(conv_out).mean(dim=2) for conv_out in x]  # mean pooling
        # x = [torch.tanh(conv_out).max(dim=2)[0] for conv_out in x]  # tanh版本的textcnn，不如relu函数
        if mask is not None:
            mask_sum = torch.sum(mask,dim=1).unsqueeze(-1) # [batch_size,1]
            x_1 = [torch.relu(conv_out).sum(dim=2) / mask_sum for conv_out in x]  # mean pooling
            x_1 = torch.cat(x_1,dim=1)  # 把列表组织成张量
            x_0 = torch.cat(x_0,dim=1)  # 把列表组织成张量
            x = x_1 + x_0   # mean 和 max pooling相加
        else:
            x_0 = torch.cat(x_0,dim=1)  # 只有max pooling
            x_1 = torch.cat(x_1,dim=1)
            x = x_0 + x_1
        x = self.fc(x)  # Fully connected layer
        return x
class TextDGCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes, num_filters):
        super(TextDGCNN, self).__init__()
        # 这里的Conv1d是same padding，也就是输出维度不会缩减
        self.convs = nn.ModuleList([
            Conv1d(input_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)

    def forward(self, x,mask=None):
        # x: [batch_size, seq_len, hidden_size]
        # mask: [batch_size,seq_len]  只有文本部分为1，mask部分为0，如果为None，代表不使用mean pooling
        x = x.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]

        x = [conv(x) for conv in self.convs]  # Apply convolutional filters
        # x = x * (mask.unsqueeze(1))

        x_0 = [torch.relu(conv_out).max(dim=2)[0] for conv_out in x]  # Max pooling
        x_1 = [torch.relu(conv_out).mean(dim=2) for conv_out in x]  # mean pooling
        # x = [torch.tanh(conv_out).max(dim=2)[0] for conv_out in x]  # tanh版本的textcnn，不如relu函数
        if mask is not None:
            # mask_sum = torch.sum(mask,dim=1).unsqueeze(-1) # [batch_size,1]
            # x_1 = [torch.relu(conv_out).sum(dim=2) / mask_sum for conv_out in x]  # mean pooling
            x_1 = [torch.relu(conv_out).mean(dim=2) for conv_out in x]  # mean pooling
            x_1 = torch.cat(x_1,dim=1)  # 把列表组织成张量
            x_0 = torch.cat(x_0,dim=1)  # 把列表组织成张量
            x = x_1 + x_0   # mean 和 max pooling相加
        else:

            x_0 = torch.cat(x_0,dim=1)  # 只有max pooling
            x_1 = torch.cat(x_1,dim=1)
            x = x_0 + x_1
        x = self.fc(x)  # Fully connected layer
        return x

if __name__ == "__main__":

    # Example usage
    batch_size = 64
    seq_len = 130
    hidden_size = 768
    output_dim = 500
    kernel_sizes = [3, 4, 5]
    num_filters = 100

    model = TextDGCNN(hidden_size, output_dim, kernel_sizes, num_filters)
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    output = model(input_data)
    print(output.shape)
