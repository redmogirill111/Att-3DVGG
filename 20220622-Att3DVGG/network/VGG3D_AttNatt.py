import torch
import torch.nn as nn
from config import Path
import torch.nn.functional as F


class VGG3D_Att(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG3D_Att, self).__init__()

        self.conv1_1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.att_at1 = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 64 * 16 * 16

        self.conv2_1 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 128 * 8 * 8

        self.conv3_1 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        # self.conv3_3 = nn.Conv3d(128, 128, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 128 * 4 * 4

        self.conv4_1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        # self.conv4_3 = nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 128 * 2 * 2

        self.conv5_1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv5_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.att_at2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        # self.conv5_3 = nn.Conv3d(256, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1))

        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(1152, 256)
        self.fc7 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # print(x.size())
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        gate = F.sigmoid(self.att_at1(conv1_1))
        output = torch.mul(conv1_2, gate)
        pool1 = self.pool1(output)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        # conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        # conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))

        gate2 = F.sigmoid(self.att_at2(conv5_1))
        output = torch.mul(conv5_2, gate2)

        # conv5_3 = self.relu(self.conv5_3(conv5_2))
        pool5 = self.pool5(output)

        # print(pool5.size())
        flat = torch.flatten(pool5, 1)

        fc6 = self.relu(self.fc6(flat))
        fc6 = self.dropout(fc6)

        fc7 = self.fc7(fc6)
        # fc7 = self.relu(self.fc7(fc6))

        # fc7 = self.fc8(fc6)
        # probs = self.softmax(fc8)

        return fc7

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = VGG3D_Att(num_classes=3)

    outputs = net.forward(inputs)
    print(outputs.size())

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, (3, 16, 112, 112), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
