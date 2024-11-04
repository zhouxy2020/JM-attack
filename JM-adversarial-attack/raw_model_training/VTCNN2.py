import torch.nn as nn
import torch
import torch.nn.functional as F

class VTCNN2(nn.Module):
    def __init__(self, dataset='128'):
        super(VTCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.drop1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), bias=False)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc_pool = nn.Linear(126, 128)
        self.dense = nn.Linear(10240, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.classfier = nn.Linear(256, 11)

    def forward(self, x):
        funcs = []
        x = x.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = self.drop1(x)
        x = F.relu(self.conv2(x)).squeeze(dim=2)
        x = self.fc_pool(x)
        x = self.drop2(x).view(x.size(0), -1)
        out3 = F.relu(self.dense(x))
        x = self.drop3(out3)
        x = self.classfier(x)
        funcs.append(out3)
        funcs.append(x)
        return funcs

