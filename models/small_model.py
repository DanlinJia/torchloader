import torch
import torch.nn as nn
import torch.nn.functional as F


class small2(nn.Module):
    def __init__(self):
        super(small2, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1000)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.avgpool(x)
        # print(1, x.shape)
        # x = x.reshape(x.size(0), -1)
        # print(1, x.shape)
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output

class small3(nn.Module):
    def __init__(self):
        super(small3, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small4(nn.Module):
    def __init__(self):
        super(small4, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output

class small5(nn.Module):
    def __init__(self):
        super(small5, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small6(nn.Module):
    def __init__(self):
        super(small6, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small7(nn.Module):
    def __init__(self):
        super(small7, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small8(nn.Module):
    def __init__(self):
        super(small8, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 32, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 32, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 32, 3, 1),
            nn.Conv2d(32, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small9(nn.Module):
    def __init__(self):
        super(small9, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 64, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 128, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output


class small10(nn.Module):
    def __init__(self):
        super(small10, self).__init__()
        self.multiconvs = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(3, 64, 3, stride=1),
            # nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 128, 3, 1),
            nn.Conv2d(128, 128, 3, 1),
        )
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1000)

    # x represents our data
    def forward(self, x):
        x = self.multiconvs(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output