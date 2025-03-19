import os
import cv2
import torch
import numpy as np
import datetime
from network import C3D_model, R2Plus1D_model, R3D_model, T2CC3D_model, DIY, C3D_AttNAtt, p3d_model, R2Plus1D_atten, \
    resnet_3d, xception_3D, mobilenetv2, shufflenetv2, squeezenet, MobileNetV3, VGG3D_AttNatt, DIY_X, F_6M3DC, Dahan_3D, \
    DenseNet_3D, GoogLeNet


def randomflip(buffer):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""
    # 以0.5的概率随机水平翻转给定图像和地面实况。

    if np.random.random() < 0.5:
        for i, frame in enumerate(buffer):
            frame = cv2.flip(buffer[i], flipCode=1)
            buffer[i] = cv2.flip(frame, flipCode=1)

    return buffer


def normalize(buffer):
    for i, frame in enumerate(buffer):
        # 修改为4通道
        # frame -= np.array([[[90.0, 98.0, 102.0]]])
        frame -= np.array([[[90.0, 98.0, 102.0, 90]]])
        buffer[i] = frame

    return buffer


def to_tensor(buffer):
    return buffer.transpose((3, 0, 1, 2))


def load_frames(file_dir):
    # 实际图片序列数据集的长和宽
    height = 128
    width = 171
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    frame_count = len(frames)
    # 修改为4通道
    # buffer = np.empty((frame_count, height, width, 3), np.dtype('float32'))
    buffer = np.empty((frame_count, height, width, 4), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        # 读取4通道
        # frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        frame = np.array(cv2.imread(frame_name, -1)).astype(np.float64)
        buffer[i] = frame

    return buffer


def crop(buffer, clip_len, crop_size):
    # 修改了此处为了屏蔽剪裁操作，避免122 * 122

    # randomly select time index for temporal jittering
    if (buffer.shape[0] - clip_len) <= 0:
        print(buffer.shape[0] - clip_len)
        print("low >= high")

    time_index = np.random.randint(buffer.shape[0] - clip_len)
    # Randomly select start indices in order to crop the video
    # 随机选择起始索引以裁剪视频
    height_index = np.random.randint(buffer.shape[1] - crop_size)
    width_index = np.random.randint(buffer.shape[2] - crop_size)

    # Crop and jitter the video using indexing. The spatial crop is performed on
    # the entire array, so each frame is cropped in the same location. The temporal
    # jitter takes place via the selection of consecutive frames
    # 使用索引裁剪和抖动视频。 空间裁剪在整个数组，所以每一帧都在同一位置裁剪。 时间的抖动通过选择连续帧来发生
    buffer = buffer[time_index:time_index + clip_len,
             height_index:height_index + crop_size,
             width_index:width_index + crop_size, :]

    return buffer


def dataset_loader(root_path):
    # 函数读取root_path,返回fnames数据集列表,labels_str标签名，labels_int标签序号
    # labels 是从root_path路径中得到的标签的列表
    # fname 是数据集的最后一层文件夹组成的列表，正常情况应该是视频样本的列表 'XX\\dataset/fire\\ALARM_CCDNIR_20220307-153158_12', 'XX\\dataset/fire\\ALARM_CCDNIR_20220307-153341_4',
    fnames, labels_str = [], []
    for label in sorted(os.listdir(root_path)):
        for fname in os.listdir(os.path.join(root_path, label)):
            fnames.append(os.path.join(root_path, label, fname))
            labels_str.append(label)
    assert len(labels_str) == len(fnames)
    print('测试样本数量: ', str(len(fnames)))
    # Prepare a mapping between the label names (strings) and indices (ints)
    label2index = {label: index for index, label in enumerate(sorted(set(labels_str)))}
    print("测试样本的标签：", str(label2index))
    # Convert the list of label names into an array of label indices
    labels_int = np.array([label2index[label] for label in labels_str], dtype=int)
    print("测试样本的序号：", str(labels_int))
    return fnames, label2index, labels_int


# 代码参考https://github.com/jfzhang95/pytorch-video-recognition/inference.py
# 1、数据集文件组织成以下结构
# dataset/fire/ALARM_CCDNIR_20220307-152638_4/00000.png
# dataset/negetive/
# dataset/smoke/
# 2、修改程序开头参数和network目录下的py网络结构文件

if __name__ == "__main__":
    ###############################################修改此处################################################################
    # 要使用全英文目录，否则cv2.imread读不出来
    root_path = r"Z:\tmp\sp_test\dataset/"
    result_file = r"Z:\tmp\sp_test\result_file.txt"
    labels_file = r'F:\THHI\program\Fire-Detection-Base-3DCNN\dataloaders\fire_labels.txt'
    modelName = 'GoogLeNet'
    # C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC F_6M3DC Dahan_3D DenseNet_3D GoogLeNet
    checkpoint = torch.load(
        # r'E:\训练记录\VGG_3D\9-4通道时空两个注意力机制-99.90\models\vgg_3D-ucf101_epoch-158_acc-0.9990.pth.tar',
        # r'E:\训练记录\对比实验\Dahan_3D\1-99.62\models\Dahan_3D-ucf101_epoch-189_acc-0.9962.pth.tar',
        # r'E:\训练记录\对比实验\DenseNet_3D\1-99.43\models\DenseNet_3D-ucf101_epoch-198_acc-0.9943.pth.tar',
        # r'E:\训练记录\对比实验\F_6M3DC\1-94.64\models\F_6M3DC-ucf101_epoch-191_acc-0.9464.pth.tar',
        r'E:\训练记录\对比实验\GoogLeNet\models\GoogLeNet-ucf101_epoch-109_acc-0.9953.pth.tar',
        map_location=lambda storage, loc: storage)
    ##############################################修改此处#################################################################
    # 图片尺寸 171 x 128
    clip_len = 16
    crop_size = 112
    num_classes = 3
    time_process = datetime.datetime.now() - datetime.datetime.now()
    time_predict = datetime.datetime.now() - datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    time_run_start = datetime.datetime.now()
    print("程序开始执行时间：", str(time_run_start))
    with open(labels_file, 'r', encoding='utf-8') as f:
        class_names = f.readlines()
        f.close()
    with open(result_file, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(
            "程序开始执行" + " " + str(modelName) + " " + "样本：" + str(root_path) + " " + "时间：" + str(time_run_start) + "\n")
    fnames, label2index, labels_int = dataset_loader(root_path)
    if not os.path.exists(result_file):
        with open(result_file, 'w', encoding='utf-8') as f:
            pass
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(
            r'F:\THHI\program\pytorch-video-recognition-ubuntu\run\run_0\models\R3D-ucf101_epoch-38.pth.tar',
            map_location=lambda storage, loc: storage)
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        checkpoint = torch.load(
            r'F:\THHI\program\pytorch-video-recognition-ubuntu\run\run_0\models\R3D-ucf101_epoch-38.pth.tar',
            map_location=lambda storage, loc: storage)
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        checkpoint = torch.load(
            r'F:\THHI\program\pytorch-video-recognition-ubuntu\run\run_0\models\R3D-ucf101_epoch-38.pth.tar',
            map_location=lambda storage, loc: storage)
    elif modelName == 'VGG':
        model = T2CC3D_model.vgg_3D(num_classes=num_classes)
    elif modelName == 'DIY':
        model = DIY.DIY(num_classes=num_classes)
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_atten.R2Plus1D(num_classes=3, layer_sizes=(3, 4, 6, 3))
    elif modelName == 'vgg_3D':
        model = T2CC3D_model.vgg_3D(num_classes=3)
    elif modelName == 'C3D_AttNAtt':
        model = C3D_AttNAtt.C3D_AttNAtt2(sample_size=112, num_classes=3,
                                         lstm_hidden_size=512, lstm_num_layers=3)
    elif modelName == 'P3D199':
        model = p3d_model.P3D199(num_classes=num_classes)
    elif modelName == 'resnet10':
        model = resnet_3d.resnet10(num_classes=num_classes)
    elif modelName == 'Xception':
        model = xception_3D.Xception(num_classes=num_classes)
    elif modelName == 'mobilenetv2':
        model = mobilenetv2.get_model(num_classes=3, sample_size=112, width_mult=1.)
    elif modelName == 'ShuffleNetV2':
        model = shufflenetv2.get_model(num_classes=3, sample_size=112, width_mult=1.)
    elif modelName == 'SqueezeNet':
        model = squeezenet.get_model(num_classes=3, sample_size=112, width_mult=1.)
    elif modelName == 'mobilenetv3_large':
        model = MobileNetV3.get_model(num_classes=3, sample_size=112, width_mult=1.)
    elif modelName == 'DIY_X':
        model = DIY_X.Xception(num_classes=3)
    elif modelName == 'VGG3D_Att':
        model = VGG3D_AttNatt.VGG3D_Att(num_classes=3)
    elif modelName == 'GoogLeNet':
        model = GoogLeNet.GoogLeNet(num_classes=3)
    elif modelName == 'F_6M3DC':
        model = F_6M3DC.F_6M3DC(num_classes=3)
    elif modelName == 'Dahan_3D':
        model = Dahan_3D.Dahan_3D(num_classes=3)
    elif modelName == 'DenseNet_3D':
        model = DenseNet_3D.DenseNet(num_init_features=64,
                                     growth_rate=32,
                                     # block_config=(6, 12, 24, 16))
                                     block_config=(32, 64, 128))
    else:
        print('再初始化一个模型吧，程序没有找到需要的模型！')
        raise NotImplementedError

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    for index in range(len(fnames)):
        print("-" * 25, str(index), "/", str(len(fnames)), "-" * 25)
        print("当前处理的文件是:", str(fnames[index]))
        allframes = load_frames(file_dir=fnames[index])
        time_run_process = datetime.datetime.now()

        # 将allframes分17帧为一个批次提取进行推理
        while (allframes.shape[0] >= 17):
            buffer = allframes[0:17, :, :, :]
            allframes = allframes[18:, :, :, :]
            buffer = crop(buffer, clip_len, crop_size)
            labels = np.array(labels_int[index])

            # # 执行数据增强
            # buffer = randomflip(buffer)
            buffer = normalize(buffer)
            # buffer = to_tensor(buffer)
            # buffer = torch.from_numpy(buffer)
            # labels = torch.from_numpy(labels)

            # 记录处理帧的时间
            time_run_predict = datetime.datetime.now()
            time_process = time_process + time_run_predict - time_run_process

            # 开始推理网络
            inputs = np.array(buffer).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            print(f"当前推理结果为{class_names[label].split(' ')[-1].strip()}")

            # 记录推理帧的时间
            time_over_predict = datetime.datetime.now()
            time_predict = time_predict + time_over_predict - time_run_predict
            with open(result_file, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                f.write(str(index) + "/" + str(len(fnames)) + "\t" + str(fnames[index]) + "\n")
                f.write("当前文件帧处理时间：" + str(time_run_predict - time_run_process) + " " + "当前文件推理时间：" + str(
                    time_over_predict - time_run_predict) + " " + "当前推理结果为：" + str(
                    class_names[label].split(' ')[-1].strip()) + "\n")
    time_run_over = datetime.datetime.now()
    print(f"处理{str(len(fnames))}个文件需要{str(time_run_over - time_run_start)}秒。")
    print(f"其中，预处理帧需要{str(time_process)}秒，推理需要{str(time_predict)}秒。")
    with open(result_file, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write("处理" + str(len(fnames)) + "个文件，使用了" + str(time_run_over - time_run_start) + "秒 " + "其中，预处理帧需要" + str(
            time_process) + "秒 " + "推理需要" + str(time_predict) + "秒" + "\n")
        f.write("--" * 30 + "文件处理结束" + "--" * 30 + "\n")
