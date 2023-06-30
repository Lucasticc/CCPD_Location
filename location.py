import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x

class PlateDetector(nn.Module):
    def __init__(self):
        super(PlateDetector, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 注意力机制
        self.attention = AttentionModule(64, 1)
        
        # 输出层
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 4) # 输出4个数字，分别代表车牌位置的左上角和右下角坐标
    
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 注意力机制
        attention_map = self.attention(x) # 生成注意力图
        x = x * attention_map # 对特征图进行加权
        
        # 展开
        x = x.view(x.size(0), -1)
        
        # 输出层
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        
        return x