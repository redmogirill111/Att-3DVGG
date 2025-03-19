import torch
import torch.nn as nn

class DIY(nn.Module):
    def __init__(self, num_classes=3):
        super(DIY, self).__init__()

        self.conv1_1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv1_2 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 1, 1))
        self.conv1_3 = nn.Conv3d(64, 64, kernel_size=(5, 1, 1), stride=1, padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))  # 64 * 16 * 16

        self.conv2_1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 1, 1))
        self.conv2_3 = nn.Conv3d(128, 128, kernel_size=(5, 1, 1), stride=1, padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))  # 128 * 8 * 8

        self.conv3_1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3_2 = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(1, 1, 1))
        self.conv3_3 = nn.Conv3d(128, 128, kernel_size=(5, 1, 1), stride=1, padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))  # 128 * 4 * 4

        self.conv4_1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv4_2 = nn.Conv3d(256, 256, kernel_size=(3, 1, 1), stride=1, padding=(1, 1, 1))
        self.conv4_3 = nn.Conv3d(256, 256, kernel_size=(5, 1, 1), stride=1, padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 128 * 2 * 2

        self.conv5_1 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv5_2 = nn.Conv3d(256, 256, kernel_size=(3, 1, 1), stride=1, padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = nn.Linear(20736, 4096)
        self.fc7 = nn.Linear(4096, 1000)
        self.fc8 = nn.Linear(1000, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax()
        self.__init_weight()

    def forward(self, x):

        conv1_1 = self.relu(self.conv1_3(self.conv1_2(self.conv1_1(x))))
        pool1 = self.pool1(conv1_1)

        conv2_1 = self.relu(self.conv2_3(self.conv2_2(self.conv2_1(pool1))))
        pool2 = self.pool2(conv2_1)

        conv3_1 = self.relu(self.conv3_3(self.conv3_2(self.conv3_1(pool2))))
        pool3 = self.pool3(conv3_1)

        conv4_1 = self.relu(self.conv4_3(self.conv4_2(self.conv4_1(pool3))))
        pool4 = self.pool4(conv4_1)

        conv5_1 = self.relu(self.conv5_2(self.conv5_1(pool4)))
        pool5 = self.pool5(conv5_1)


        flat = torch.flatten(pool5, 1)
        # x = x.view(-1, 64)

        fc6 = self.relu(self.fc6(flat))
        fc7 = self.relu(self.fc7(fc6))
        fc7 = self.dropout(fc7)
        fc7 = self.fc8(fc7)

        return fc7

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

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
    inputs = torch.rand(1, 3, 16, 224, 224)
    net = DIY(num_classes=3)
    # print("--"*80)
    # print(net)
    # print("--" * 80)

    # from torch.autograd import Variable
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # inputs = Variable(inputs, requires_grad=False).to(device)
    # net.to(device)

    outputs = net.forward(inputs)
    print(outputs.size())

    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(net, (3, 16, 224, 224), as_strings=True,
    #                                          print_per_layer_stat=False, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))