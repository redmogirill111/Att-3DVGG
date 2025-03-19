import os
import torch
import numpy as np
from network import C3D_model, R2Plus1D_model, R3D_model, T2CC3D_model, DIY, C3D_AttNAtt, p3d_model, R2Plus1D_atten, \
    resnet_3d, xception_3D, mobilenetv2, shufflenetv2, squeezenet, MobileNetV3, VGG3D_AttNatt, DIY_X, F_6M3DC, Dahan_3D, \
    DenseNet_3D, GoogLeNet
import cv2

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop3(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def center_crop4(frame):
    frame = frame[8:120, 30:142, :, :]
    return np.array(frame).astype(np.uint8)


def main():
    ###############################################修改此处################################################################
    # 要使用全英文目录，否则cv2.imread读不出来
    root_path = r"D:\tmp\video_dataset_test\VisiFire-20"
    det_path = r"D:\tmp\video_dataset_test\VisiFire-20"
    result_file = r"D:\tmp\video_dataset_test\VisiFire-20\allwithoutlabel.txt"
    labels_file = r'F:\THHI\program\Fire-Detection-Base-3DCNN\dataloaders\fire_labels.txt'
    modelName = 'DIY_X'
    # C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC F_6M3DC Dahan_3D DenseNet_3D GoogLeNet
    checkpoint = torch.load(
        r'E:\训练记录\第二篇论文\20220808-192739_DIY_X_0.0001__THHICV2\models\DIY_X-ucf101_epoch-23_acc-0.9934.pth.tar',
        # r'E:\训练记录\对比实验\Dahan_3D\1-99.62\models\Dahan_3D-ucf101_epoch-189_acc-0.9962.pth.tar',
        # r'E:\训练记录\对比实验\DenseNet_3D\1-99.43\models\DenseNet_3D-ucf101_epoch-198_acc-0.9943.pth.tar',
        # r'E:\训练记录\对比实验\F_6M3DC\1-94.64\models\F_6M3DC-ucf101_epoch-191_acc-0.9464.pth.tar',
        # r'E:\训练记录\对比实验\20220625-100925_R2Plus1D_0.01__THHICV2\models\R2Plus1D-ucf101_epoch-180_acc-0.9276.pth.tar',
        map_location=lambda storage, loc: storage)
    cut_frame = 1
    num_classes = 3
    tongdao = 3
    ##############################################修改此处#################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device being used:", device)

    with open(labels_file, 'r', encoding='utf-8') as f:
        class_names = f.readlines()
        f.close()
    with open(result_file, 'w', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        pass
    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
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

    current_video = 1
    import datetime
    starttime = datetime.datetime.now()
    for root_path, dirs, files in os.walk(root_path):  # 这里就填文件夹目录就可以了
        for file in files:
            # 获取文件路径
            if (('.avi' in file) or ("mp4" in file) or (".AVI" in file) or (".MP4" in file)):
                video_full_path = os.path.join(root_path, file)
                video = cv2.VideoCapture(video_full_path)
                current_frame = 1
                file_sp_no = 0
                # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                totalframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                clip = []
                while (True):
                    ret, frame = video.read()

                    current_frame = current_frame + 1
                    if ret is False:
                        print("没有从", video_full_path, "读到帧,可能该文件快结束")
                        video.release()
                        break

                    if current_frame % cut_frame == 0:

                        if tongdao == 3:
                            tmp_ = center_crop3(cv2.resize(frame, (171, 128)))
                            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])

                        else:
                            # 截取帧
                            rgb = frame[:, :1920]
                            nif = frame[:, 1920:]

                            # # RongHe
                            # ronghelv = 0.5
                            # frame = rgb * ronghelv + nif * (1 - ronghelv)

                            # 4通道
                            b_rgb, g_rgb, r_rgb = cv2.split(rgb)
                            b_nif, g_nif, r_nif = cv2.split(nif)
                            chanel_4 = cv2.merge((b_rgb, g_rgb, r_rgb, r_nif))
                            frame = chanel_4
                            tmp_ = center_crop4(cv2.resize(frame, (171, 128)))

                            # 4通道正Z化
                            tmp = tmp_ - np.array([[[90.0, 98.0, 102.0, 90]]])

                            # 灰度化 灰度化要放在最后一部 不然就报错
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.merge((frame, frame, frame))

                        clip.append(tmp)
                        if len(clip) == 16:
                            inputs = np.array(clip).astype(np.float32)
                            inputs = np.expand_dims(inputs, axis=0)
                            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
                            inputs = torch.from_numpy(inputs)
                            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
                            with torch.no_grad():
                                outputs = model.forward(inputs)

                            probs = torch.nn.Softmax(dim=1)(outputs)
                            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

                            with open(result_file, 'a',
                                      encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                                f.write("mv" + " " + str(video_full_path) + " " + os.path.join(det_path, str(
                                    class_names[label].split(' ')[-1].strip())) + "\n")
                            print("mv" + " " + str(video_full_path) + " " + os.path.join(det_path, str(
                                class_names[label].split(' ')[-1].strip())) + "\n")

                            # with open(result_file, 'a',
                            #           encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                            #     f.write("文件：" + str(video_full_path) + "\t" + "预测：" + str(
                            #         class_names[label].split(' ')[-1].strip()) + "\t准确率" + str(
                            #         "%.4f" % probs[0][label]) + "\n")
                            # if "fire" in video_full_path:
                            #     if str(class_names[label].split(' ')[-1].strip()) != "fire":
                            #         with open(result_file, 'a', encoding='utf-8') as f:
                            #             f.write("mv" + " " + str(video_full_path) + "\n")
                            #     else:
                            #         pass
                            # if "negetive" in video_full_path:
                            #     if str(class_names[label].split(' ')[-1].strip()) != "negetive":
                            #         with open(result_file, 'a', encoding='utf-8') as f:
                            #             f.write("error" + "&" + str(video_full_path) + "\n")
                            #     else:
                            #         pass
                            #
                            # if "smoke" in video_full_path:
                            #     if str(class_names[label].split(' ')[-1].strip()) != "smoke":
                            #         with open(result_file, 'a', encoding='utf-8') as f:
                            #             f.write("error" + "&" + str(video_full_path) + "\n")
                            #     else:
                            #         pass

                            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 1)
                            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                        (0, 0, 255), 1)
                            clip.pop(0)

                        cv2.imshow('result', frame)
                        cv2.waitKey(10)
                        file_sp_no = file_sp_no + 1
                        print("总进度：", str(current_video), "/", len(files), "/", "?", " 当前文件进度：", current_frame, "/",
                              totalframes)
                current_video = current_video + 1
    endtime = datetime.datetime.now()
    # print(endtime - starttime).seconds
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
