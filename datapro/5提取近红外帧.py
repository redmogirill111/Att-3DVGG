import os
import cv2

'''
!!!!!中文目录！！！
cut_frame = 2  # 多少帧截一次，自己设置就行
root_path 为要处理的目录的，上级目录路径
sou_name 为要处理的目录名称，在root_path的里面，名称不能包含root_path的连续字符
det_name 保存的位置的目录名称
'''

cut_frame = 1  # 多少帧截一次，自己设置就行
root_path = r"F:\dataset\huoyanshujvji11111111\20220313\2-4/"
sou_name = "YuanMp4-01"
det_name = "Nif_only-01"
resize_height = 128
resize_width = 171

current_video = 1
for root_path, dirs, files in os.walk(os.path.join(root_path, sou_name)):  # 这里就填文件夹目录就可以了
    for file in files:
        # 获取文件路径
        if (('.avi' in file) or ("mp4" in file) or (".AVI" in file) or (".MP4" in file)):
            video_full_path = os.path.join(root_path, file)
            video = cv2.VideoCapture(video_full_path)
            current_frame = 1
            file_sp_no = 0
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            totalframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            while (True):
                ret, frame = video.read()
                current_frame = current_frame + 1
                if ret is False:
                    # print("没有从", video_full_path, "读到帧,可能该文件快结束")
                    video.release()
                    break

                if current_frame % cut_frame == 0:
                    save_path = os.path.join(root_path.replace(sou_name, det_name), file[:-4])
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # frame_name = "0000" + str(file_sp_no) + '.jpg'
                    # 写入4通道
                    frame_name = "0000" + str(file_sp_no) + '.png'

                    # 截取帧
                    # rgb = frame[:, :1920]
                    nif = frame[:, 1920:]

                    # # RongHe
                    # ronghelv = 0.5
                    # frame = rgb * ronghelv + nif * (1 - ronghelv)

                    # # 4通道
                    # b_rgb, g_rgb, r_rgb = cv2.split(rgb)
                    # b_nif, g_nif, r_nif = cv2.split(nif)
                    # chanel_4 = cv2.merge((b_rgb,g_rgb,r_rgb,r_nif))
                    # frame = chanel_4

                    # 3通道转成单通道

                    frame = cv2.cvtColor(nif, cv2.COLOR_BGR2GRAY)

                    # 对帧变换尺寸
                    if (frame_height != resize_height) or (frame_width != resize_width):
                        frame = cv2.resize(frame, (resize_width, resize_height))
                    img_name_full = os.path.join(save_path, frame_name)
                    cv2.imwrite(img_name_full, img=frame)
                    file_sp_no = file_sp_no + 1
                    print("总进度：", str(current_video), "/", len(files), "/", "?", " 当前文件进度：", current_frame, "/",
                          totalframes)
            video.release()
            if file_sp_no < 19:
                print(save_path, " 该视频帧数量不足19！")
            current_video = current_video + 1
