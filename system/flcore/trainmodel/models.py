import torch.nn.functional as F
from torch import nn
batch_size = 10


# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        rep = self.base(x)
        out = self.head(rep)

        return out

class CNN1D(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2, num_classes=10):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # UNSW 和 KDD 通用

    def forward(self, x):
        x = x.unsqueeze(1)  # 增加一个维度以适配 1D CNN 的输入要求
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.global_maxpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x
