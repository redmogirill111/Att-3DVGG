import torch
import torch.nn as nn
import torch.nn.functional as F


class Depthwise_Separable2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depthwise_Separable2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

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


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride, bias=False)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, dilation))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, dilation))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, dilation))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d(planes, planes, 3, 2))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv3d(planes, planes, 3, 1))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
        else:
            skip = inp

        x = x + skip

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """

    def __init__(self, num_classes=3):
        super(Xception, self).__init__()

        middle_block_dilation = 1
        exit_block_dilations = (1, 2)
        # exit_block_dilations = (2, 4)
        self.relu = nn.ReLU(inplace=True)

        # Entry flow
        self.conv1 = ConvBlock(3, 32, kernel_size=3, stride=1, bias=False)

        self.conv2 = ConvBlock(32, 64, kernel_size=4, stride=1, bias=False)

        self.conv3 = ConvBlock(64, 64, kernel_size=3, stride=1, bias=False)

        self.block1 = Block(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, start_with_relu=False, grow_first=True)
        self.block3 = Block(256, 384, reps=2, stride=1,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4 = Block(384, 384, reps=3, stride=1, dilation=middle_block_dilation,
                            start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(384, 512, reps=2, stride=1, dilation=exit_block_dilations[0],
                             start_with_relu=True, grow_first=False, is_last=True)

        self.conv4 = SeparableConv3d(512, 768, 3, stride=1, dilation=exit_block_dilations[1])

        self.conv5 = SeparableConv3d(768, 768, 3, stride=1, dilation=exit_block_dilations[1])

        self.conv6 = SeparableConv3d(768, 1024, 3, stride=1, dilation=exit_block_dilations[1])

        self.fc = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

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

        # Entry flow
        x = self.conv1(x)
        x = F.max_pool3d(x, kernel_size=3, stride=(1, 2, 2))
        x = self.conv2(x)
        x = F.avg_pool3d(x, kernel_size=3, stride=1)
        x = self.conv3(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    inputs = torch.rand(1, 4, 16, 112, 112)
    net = Xception(num_classes=3)
    # print("--"*80)
    # print(net)
    # print("--" * 80)
    outputs = net.forward(inputs)
    print(outputs.size())
    macs, params = get_model_complexity_info(net, (4, 16, 112, 112), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
