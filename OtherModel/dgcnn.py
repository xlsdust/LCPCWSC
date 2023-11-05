import torch
from torch import nn
from .same_padding_conv12d import Conv1d
class DilatedGatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_rate=1, skip_connect=True, dropout_gate=0.1):
        super(DilatedGatedConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.skip_connect = skip_connect
        self.dropout_gate = dropout_gate
        # 这里的Conv1d为same padding，输入和输出维度一致
        self.conv = Conv1d(self.in_channels, self.out_channels * 2, self.kernel_size, dilation=self.dilation_rate)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_gate)
        # 如果使用残差连接
        if self.skip_connect:
            self.conv1d_1x1 = Conv1d(self.in_channels, self.out_channels, 1, dilation=1)
    def forward(self, input,batch_text_mask):
        """
        :param input: [batch_size,seq_len,hidden_size]
        :param batch_text_mask: [batch_size,seq_len]
        :return:
        """
        input = input.permute(0,2,1)    # 转换为 [batch_size,hidden_size,seq_len]
        mask = batch_text_mask.unsqueeze(1).float() # 转换为[batch_size,1,seq_len]
        input = input * mask
        x = self.conv(input)    # 卷积的输出 [batch_size，out_channels * 2，seq_len]
        # 将输出张量x切分为两部分，分别赋给x和g。其中，x的维度为[batch_size,out_channels,seq_len]，g的维度为[batch_size,out_channels,seq_len:]
        x, g = x[..., :self.out_channels,:], x[..., self.out_channels:,:]
        # # 在训练阶段对g进行扰动
        # if self.drop_gate is not None:
        #     g = torch.nn.functional.dropout(g, self.drop_gate, training=self.training)
        # x [batch_size,out_channels,seq_len]
        # g [batch_size,out_channels,seq_len]
        # 对g应用sigmoid激活函数。
        g = torch.sigmoid(g)
        # 如果使用残差连接
        if self.skip_connect:
            # 如果self.o_dim不等于xo的最后一个维度，将xo作为输入传递给之前创建的conv1d_1x1层，并调用self.reuse方法，对conv1d_1x1层进行复用
            if self.out_channels != input.size(-1):
                input = self.conv1d_1x1(input)
            return ((input * (1 - g) + x * g) * mask).permute(0,2,1)
        else:
            return (x * g * mask).permute(0,2,1)