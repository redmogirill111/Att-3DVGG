import cv2
import numpy as np
import matplotlib.pyplot as plt

##################################################################################
# 输入视频
# 将每个视频帧裁分解为 直方图
# !!!!!中文目录！！！
##################################################################################
FileName = r"D:\mp4\nege2.avi"
cap = cv2.VideoCapture(FileName)  # 原视频位置
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = cv2.VideoWriter(r"D:\mp4\fire.avi", fourcc, 15, (int(w), int(h)), True)
i = 0
while (True):
    ret, frame = cap.read()
    if not ret:  # or count >= EndNum:
        print('not res , not image')
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # index = len(np.where(gray == 255)[0])
    # print(index)
    # img2 = cv2.resize(frame, (int(w), int(h)))
    # cv2.imshow('a', img2)
    # cv2.waitKey(1)
    # # 直方图
    # plt.hist(gray.flatten(), 256)
    # plt.show()
    # plt.close('all')

    chans = cv2.split(frame)
    colors = ("b", "g", "r")
    # plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    # fig = plt.figure()
    # fig.savefig("D:\mp4/smoke/" + str(i) + ".jpg")
    i = i + 1

    plt.savefig("D:\mp4/nege2/" + str(i) + ".jpg")
    # plt.show()
cap.release()
