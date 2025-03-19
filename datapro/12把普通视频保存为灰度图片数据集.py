import os
import cv2

##################################################################################
# 输入视频
# 将普通可见光视频画面保存为灰度图
# !!!!!中文目录！！！
# cut_frame = 2  # 多少帧截一次，自己设置就行
# root_path 为要处理的目录的，上级目录路径
# sou_name 为要处理的目录名称，在root_path的里面，名称不能包含root_path的连续字符
# det_name 保存的位置的目录名称
##################################################################################

cut_frame = 3  # 多少帧截一次，自己设置就行
root_path = r"D:\tmp/"
sou_name = "zhenghe"
det_name = r"zhenghe-pro"
resize_height = 128
resize_width = 171

current_video = 1
for root_path, dirs, files in os.walk(os.path.join(root_path, sou_name)):  # 这里就填文件夹目录就可以了
    for file in files:
        # 获取文件路径
        if (('.avi' in file) or ("mp4" in file) or (".AVI" in file) or (".MP4" in file)):
            video_full_path = os.path.join(root_path, file)
            video_left = cv2.VideoCapture(video_full_path)
            current_frame = 1
            file_sp_no = 0
            frame_count = int(video_left.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(video_left.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
            totalframes = int(video_left.get(cv2.CAP_PROP_FRAME_COUNT))

            while (True):
                ret, frame = video_left.read()
                current_frame = current_frame + 1
                if ret is False:
                    # print("没有从", video_full_path, "读到帧,可能该文件快结束")
                    video_left.release()
                    break

                if current_frame % cut_frame == 0:
                    save_path_left = os.path.join(root_path.replace(sou_name, det_name), file[:-4]).replace(
                        "ALARM_CCDNIR", "Left")

                    if not os.path.exists(save_path_left):
                        os.makedirs(save_path_left)

                    frame_name = "0000" + str(file_sp_no) + '.jpg'

                    # 截取帧
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # 对帧变换尺寸
                    if (frame_height != resize_height) or (frame_width != resize_width):
                        rgb = cv2.resize(rgb, (resize_width, resize_height))
                    img_name_full_left = os.path.join(save_path_left, frame_name)

                    cv2.imwrite(img_name_full_left, img=rgb)

                    file_sp_no = file_sp_no + 1
                    print("总进度：", str(current_video), "/", len(files), "/", "?", " 当前文件进度：", current_frame, "/",
                          totalframes)
            video_left.release()
            current_video = current_video + 1
