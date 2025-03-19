import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_model_summary import summary


class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size=(1, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(inplanes, inplanes, kernel_size=(1, 5, 5), stride=1, padding=(1, 2, 2))
        self.conv4 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = x

        x1 = self.conv1(x0)
        x2 = self.conv2(x0)

        x3 = x1 + x2
        x4 = self.relu(self.conv4(x3))
        return x4


class F_6M3DC(nn.Module):
    """
    The DenseNet_3D.
    """

    def __init__(self, num_classes, pretrained=False):
        super(F_6M3DC, self).__init__()

        self.block1 = Block(4, 16, stride=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.block2 = Block(16, 16, stride=1)
        self.bn2 = nn.BatchNorm3d(16)

        self.block3 = Block(16, 32, stride=1)
        self.bn3 = nn.BatchNorm3d(32)

        self.block4 = Block(32, 32, stride=1)
        self.bn4 = nn.BatchNorm3d(32)

        self.block5 = Block(32, 64, stride=1)
        self.bn5 = nn.BatchNorm3d(64)

        self.block6 = Block(64, 64, stride=1)
        self.bn6 = nn.BatchNorm3d(64)

        self.block7 = Block(64, 128, stride=1)

        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def forward(self, x):
        x0 = x
        b1 = self.block1(x0)
        b1 = self.bn1(b1)

        b2 = self.block2(b1)
        b2 = self.bn2(b2)

        b3 = self.block3(b2)
        b3 = self.bn3(b3)

        b4 = self.block4(b3)
        b4 = self.bn4(b4)

        b5 = self.block5(b4)
        b5 = self.bn5(b5)

        b6 = self.block6(b5)
        b6 = self.bn6(b6)

        b7 = self.block7(b6)
        gam = F.adaptive_avg_pool3d(b7, (1, 1, 1))
        gam = torch.flatten(gam, 1)
        f8 = self.fc(gam)
        s9 = self.softmax(f8)
        return s9

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
    inputs = torch.rand(1, 4, 16, 112, 112)
    net = F_6M3DC(num_classes=3, pretrained=False)
    # print("--"*80)
    # print(net)
    # print("--" * 80)
    outputs = net.forward(inputs)
    print(outputs.size())

    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, (4, 16, 112, 112), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # pytorch_total_params = sum(p.numel() for p in net.parameters())
    # trainable_pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #
    # print('Total - ', pytorch_total_params)
    # print('Trainable - ', trainable_pytorch_total_params)
