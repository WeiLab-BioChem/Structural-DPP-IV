import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.filter_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        self.embedding_dim = 21  # the MGF process dim
        filter_num = 64
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, self.embedding_dim)) for fsz in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)

        self.classification = nn.Sequential(
            nn.Linear(len(self.filter_sizes) * filter_num, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.cuda()
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        output = self.classification(x)

        return output, x
