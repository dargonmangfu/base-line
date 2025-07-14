import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalLSTM(nn.Module):
    """
    双向LSTM网络 - 使用LSTM单元
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 batch_first=True):
        super(BiDirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 输出层 - 双向LSTM输出维度为 hidden_size * 2
        self.output_layer = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入序列 (batch_size, seq_len, input_size) 如果batch_first=True
        Returns:
            output: 输出序列 (batch_size, seq_len, input_size)
            lstm_output: 原始LSTM输出 (batch_size, seq_len, hidden_size * 2)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 应用输出层到每个时间步
        output = self.output_layer(lstm_out)
        
        return output, lstm_out
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        初始化隐藏状态和细胞状态
        """
        # 双向LSTM需要 2 * num_layers
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return (h0, c0)


