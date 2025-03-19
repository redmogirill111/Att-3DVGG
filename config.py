import socket
import torch

# 适配多台不同机器调用
if socket.gethostname() == "THHICV":
    runName = "测试数据集8月8日"
    learnRate = 0.0001
    weightDecay = 1e-4
    gamma = 0.8
    numWorker = 4  # 数据加载使用的进程数量 一般为GPU*4
    batchSize = 10
    modelName = 'vgg_3D'
    # 待测试 Xception mobilenetv2 ShuffleNetV2 SqueezeNet MobileNetV3 GoogLeNet C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D
    # t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC Dahan_3D DenseNet_3D
    epochAll = 200
    epochResume = 0  # 默认为0，如果要恢复训练则更改
    useTest = True  # 训练时使用测试集
    useLMDB = False
    testInterval = 1  # 每隔testInterval个epoch在测试集上测试
    channel = 3  # 数据集输入的通道,只支持3、4通道输入
    numClasses = 3
    dataset_dir = r'Z:\tmp\sp_test'
    # dataset_dir = r'Z:\tmp\sp_test'
    resumePth_path = r"F:\THHI\program\Fire-Detection-Base-3DCNN\run\20220807-113606_vgg_3D_0.001__THHICV\models\vgg_3D" \
                     r"-ucf101_epoch-34_acc-0.9703.pth.tar "
    modelInit_path = r'data/vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar'
    lmdb_train = r"data/train.lmdb"
    lmdb_val = r"data/val.lmdb"
    lmdb_test = r"data/test.lmdb"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("config:当前执行程序的设备是THHICV")

elif socket.gethostname() == "THHICV2":
    runName = "230227_shift1/8_2Seg-3通道2data2_jpg"
    learnRate = 0.0001
    weightDecay = 0.00001
    gamma = 1
    numWorker = 4  # 数据加载使用的进程数量 一般为GPU*4
    batchSize = 12
    modelName = 'TSM_WITH_ACTION'
    # 待测试 Xception mobilenetv2 ShuffleNetV2 SqueezeNet MobileNetV3 GoogLeNet C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D
    # t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC Dahan_3D DenseNet_3D vgg11_3d TSM
    epochAll = 200
    epochResume = 0  # 默认为0，如果要恢复训练则更改
    useTest = True  # 训练时使用测试集
    useLMDB = False
    testInterval = 1  # 每隔testInterval个epoch在测试集上测试
    channel = 3  # 数据集输入的通道,只支持3、4通道输入
    numClasses = 3
    dataset_dir = r'G:\program\date\2paper\2data2_jpg'
    # dataset_dir = r'C:\Users\THHICV\Documents\ProgramFiles\data2_jpg_cafen'
    # dataset_dir = r'G:\program\date\2paper\34tongdaobeijingcaifen'
    # dataset_dir = r'G:\program\date\2paper\2data2_jpg'
    resumePth_path = r"F:\THHI\program\Fire-Detection-Base-3DCNN\run\20220807-113606_vgg_3D_0.001__THHICV\models\vgg_3D" \
                     r"-ucf101_epoch-34_acc-0.9703.pth.tar "
    modelInit_path = r'data/vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar'
    lmdb_train = r"G:\program\date\2paper\44tongdaobeijingcaifenlmdb\train.lmdb"
    lmdb_val = r"G:\program\date\2paper\44tongdaobeijingcaifenlmdb\val.lmdb"
    lmdb_test = r"G:\program\date\2paper\44tongdaobeijingcaifenlmdb\test.lmdb"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("config:当前执行程序的设备是V")

elif (socket.gethostname()) > 'gpu':
    runName = "测试数据集8月8日"
    learnRate = 0.0001
    weightDecay = 1e-4
    gamma = 0.8
    numWorker = 4  # 数据加载使用的进程数量 一般为GPU*4
    batchSize = 10
    modelName = 'vgg_3D'
    # 待测试 Xception mobilenetv2 ShuffleNetV2 SqueezeNet MobileNetV3 GoogLeNet C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D
    # t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC Dahan_3D DenseNet_3D
    epochAll = 200
    epochResume = 0  # 默认为0，如果要恢复训练则更改
    useTest = True  # 训练时使用测试集
    useLMDB = False
    testInterval = 1  # 每隔testInterval个epoch在测试集上测试
    channel = 4  # 数据集输入的通道,只支持3、4通道输入
    numClasses = 3
    # dataset_dir = r'Y:\8saixuan_jpg'
    dataset_dir = r'Z:\tmp\sp_test'
    resumePth_path = r"F:\THHI\program\Fire-Detection-Base-3DCNN\run\20220807-113606_vgg_3D_0.001__THHICV\models\vgg_3D" \
                     r"-ucf101_epoch-34_acc-0.9703.pth.tar "
    modelInit_path = r'data/vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar'
    lmdb_train = r"data/train.lmdb"
    lmdb_val = r"data/val.lmdb"
    lmdb_test = r"data/test.lmdb"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("config:当前执行程序的设备是GPU")

else:
    runName = "测试数据集8月8日"
    learnRate = 0.0001
    weightDecay = 1e-4
    gamma = 0.8
    numWorker = 4  # 数据加载使用的进程数量 一般为GPU*4
    batchSize = 10
    modelName = 'vgg_3D'
    # 待测试 Xception mobilenetv2 ShuffleNetV2 SqueezeNet MobileNetV3 GoogLeNet C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D
    # t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC Dahan_3D DenseNet_3D
    epochAll = 200
    epochResume = 0  # 默认为0，如果要恢复训练则更改
    useTest = True  # 训练时使用测试集
    useLMDB = True
    testInterval = 1  # 每隔testInterval个epoch在测试集上测试
    channel = 3  # 数据集输入的通道,只支持3、4通道输入
    numClasses = 3
    # dataset_dir = r'Y:\8saixuan_jpg'
    dataset_dir = r'Z:\tmp\sp_test'
    resumePth_path = r"F:\THHI\program\Fire-Detection-Base-3DCNN\run\20220807-113606_vgg_3D_0.001__THHICV\models\vgg_3D" \
                     r"-ucf101_epoch-34_acc-0.9703.pth.tar "
    modelInit_path = r'data/vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar'
    lmdb_train = r"data/train.lmdb"
    lmdb_val = r"data/val.lmdb"
    lmdb_test = r"data/test.lmdb"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("config:当前执行程序的设备是其他")
