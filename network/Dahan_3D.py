import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, inplanes, planes, padding):
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), padding=padding)
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):
        x0 = x
        c1 = self.conv1(x0)
        p2 = self.pool(c1)
        return p2


class Dahan_3D(nn.Module):
    """
    The DenseNet_3D.
    """

    def __init__(self, num_classes=3, pretrained=False):
        super(Dahan_3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(32768, 1024)
        self.fc8 = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        # print("1", x.shape)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        # print("2", x.shape)

        x = self.relu(self.conv3a(x))
        # x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        # print("3", x.shape)

        x = self.relu(self.conv4a(x))
        # x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        # print("4", x.shape)

        x = self.relu(self.conv5a(x))
        # x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        # print("5", x.shape)

        x = x.view(-1, 8192 * 4)
        # print("6", x.shape)
        x = self.relu(self.fc6(x))
        # print("7", x.shape)
        x = self.dropout(x)
        logits = self.fc8(x)
        # print("8", logits.shape)
        # probs = self.softmax(logits)
        # print("9", probs.shape)
        return logits

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()


if __name__ == "__main__":
    inputs = torch.rand(8, 4, 16, 224, 224)
    net = Dahan_3D(num_classes=3, pretrained=False)

    from torch.autograd import Variable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = Variable(inputs, requires_grad=False).to(device)
    net.to(device)

    # print("--"*80)
    # print(net)
    # print("--" * 80)
    outputs = net.forward(inputs)
    print(outputs.size())
    #
    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, (4, 16, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
