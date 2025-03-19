import TSM_Att.models as TSN_model
from TSM_Att.spatial_transforms import *

if __name__ == "__main__":
    inputs = torch.rand(3, 3, 16, 224, 224)
    net = TSN_model.TSN(3, 16, 'RGB',
                        is_shift=False,
                        partial_bn=True,
                        base_model="resnet50",
                        pretrain=False,
                        shift_div=4,
                        dropout=0.5,
                        img_feature_dim=224)
    # print("--"*80)
    # print(net)
    # print("--" * 80)

    from torch.autograd import Variable

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = Variable(inputs, requires_grad=False).to(device)
    net.to(device)

    outputs = net.forward(inputs)
    print(outputs.size())

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, (3, 16, 224, 224), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
