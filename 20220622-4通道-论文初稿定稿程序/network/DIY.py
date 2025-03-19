import torch
import torch.nn as nn
import torch.nn.functional as F


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.pointwise = nn.Conv3d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        # print('s_in', x.shape)
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        # print('s_out', x.shape)

        return x


class DIY(nn.Module):
    """
    The DIY network.
    """

    def __init__(self, num_classes):
        super(DIY, self).__init__()

        self.conv0 = nn.Conv3d(4, 32, kernel_size=(3, 3, 3), padding=(0, 0, 0), stride=(1, 1, 1))
        self.bn0 = nn.BatchNorm3d(32)
        self.pool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.conv1 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(0, 0, 0), stride=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        self.sconv2_1 = SeparableConv3d(64, 128)
        self.sconv2_2 = SeparableConv3d(128, 128)
        self.sconv2_ = nn.Conv3d(64, 128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.sconv3_1 = SeparableConv3d(128, 128)
        self.sconv3_2 = SeparableConv3d(128, 128)
        self.sconv3_ = nn.Conv3d(128, 128, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.sconv4_1 = SeparableConv3d(128, 256)
        self.sconv4_2 = SeparableConv3d(256, 256)
        self.sconv4_ = nn.Conv3d(128, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.sconv5_1 = SeparableConv3d(256, 256)
        self.sconv5_2 = SeparableConv3d(256, 256)
        self.sconv5_ = nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.sconv6 = nn.Conv3d(256, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)

        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 3*3*3卷积
        conv0 = self.conv0(x)
        conv0 = self.bn0(conv0)
        conv0 = self.relu(conv0)
        conv0 = self.pool0(conv0)

        conv1 = self.conv1(conv0)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)
        conv1 = self.pool1(conv1)

        conv2 = self.sconv2_1(conv1)
        conv2 = self.sconv2_2(conv2)
        conv2_ = self.sconv2_(conv1)
        block2 = conv2 + conv2_

        conv3 = self.sconv3_1(block2)
        conv3 = self.sconv3_2(conv3)
        conv3_ = self.sconv3_(block2)
        block3 = conv3 + conv3_

        conv4 = self.sconv4_1(block3)
        conv4 = self.sconv4_2(conv4)
        conv4_ = self.sconv4_(block3)
        block4 = conv4 + conv4_

        conv5 = self.sconv5_1(block4)
        conv5 = self.sconv5_2(conv5)
        conv5_ = self.sconv5_(block4)
        block5 = conv5 + conv5_

        conv6 = self.sconv6(block5)
        conv6 = F.adaptive_avg_pool3d(conv6, (1, 1, 1))

        x = self.dropout(conv6)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    inputs = torch.rand(1, 4, 16, 112, 112)
    net = DIY(num_classes=3)
    # print("--"*80)
    # print(net)
    # print("--" * 80)
    outputs = net.forward(inputs)
    print(outputs.size())
    macs, params = get_model_complexity_info(net, (4, 16, 112, 112), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # # 打印网络模型结构 https://www.freesion.com/article/51191315925/
    # from torchviz import make_dot
    # g = make_dot(outputs)
    # g.view(r"C:\Users\THHICV\Downloads/DIY.gv")

    # # 使用netron可视化模型
    # import torch.onnx
    # import netron
    # onnx_path = r"C:\Users\THHICV\Downloads/DIY_net.onnx"
    # torch.onnx.export(net, inputs, onnx_path)
    # netron.start(onnx_path)
