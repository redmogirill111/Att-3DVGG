import os
import torch
import numpy as np
from network import C3D_model, DIY_X
import cv2

torch.backends.cudnn.benchmark = True

# root_path = r"F:\dataset\huoyanshujvji11111111\20220313\2-4\YuanMP4/"
root_path = r"G:\program\date\YuanMP4/"
global result_file
result_file = r"G:\program\Fire-Detection-Base-3DCNN\run\run_0\result_file.txt"
cut_frame = 3


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    global root_path
    global resize_height
    global resize_width

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open(r'../dataloaders/fire_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    with open(result_file, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        pass

    # model = R3D_model.R3DClassifier(num_classes=3, layer_sizes=(2, 2, 2, 2))
    model = DIY_X.Xception(num_classes=3)
    checkpoint = torch.load(r'G:\program\Fire-Detection-Base-3DCNN\data\DIY_X-ucf101_epoch-105_acc-0.9981.pth.tar',
                            map_location=lambda storage, loc: storage)
    # checkpoint = torch.load(r'F:\THHI\program\pytorch-video-recognition-ubuntu\run\run_0\models\R3D-ucf101_epoch-38.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    current_video = 1
    import datetime
    starttime = datetime.datetime.now()
    for root_path, dirs, files in os.walk(root_path):  # 这里就填文件夹目录就可以了
        for file in files:
            # 获取文件路径
            if ('.avi' in file):
                video_full_path = os.path.join(root_path, file)
                video = cv2.VideoCapture(video_full_path)
                current_frame = 1
                file_sp_no = 0
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                totalframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                clip = []
                while (True):
                    ret, frame = video.read()
                    current_frame = current_frame + 1
                    if ret is False:
                        # print("没有从", video_full_path, "读到帧,可能该文件快结束")
                        video.release()
                        break

                    if current_frame % cut_frame == 0:
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

                        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                        # 4通道正Z化
                        # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0, 90]]])
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
                            with open(result_file, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                                f.write(str(video_full_path) + "&" + str(
                                    class_names[label].split(' ')[-1].strip()) + "&" + str(
                                    "%.4f" % probs[0][label]) + "\n")
                            if "fire" in video_full_path:
                                if str(class_names[label].split(' ')[-1].strip()) != "fire":
                                    with open(result_file, 'a') as f:
                                        f.write("error" + "&" + str(video_full_path) + "\n")
                                else:
                                    pass
                            if "negetive" in video_full_path:
                                if str(class_names[label].split(' ')[-1].strip()) != "negetive":
                                    with open(result_file, 'a') as f:
                                        f.write("error" + "&" + str(video_full_path) + "\n")
                                else:
                                    pass

                            if "smoke" in video_full_path:
                                if str(class_names[label].split(' ')[-1].strip()) != "smoke":
                                    with open(result_file, 'a') as f:
                                        f.write("error" + "&" + str(video_full_path) + "\n")
                                else:
                                    pass

                            # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            #             (0, 0, 255), 1)
                            # cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            #             (0, 0, 255), 1)
                            clip.pop(0)

                        # cv2.imshow('result', frame)

                        file_sp_no = file_sp_no + 1
                        print("总进度：", str(current_video), "/", len(files), "/", "?", " 当前文件进度：", current_frame, "/",
                              totalframes)
                current_video = current_video + 1
    endtime = datetime.datetime.now()
    print(endtime - starttime).seconds
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
