import os
import cv2

##################################################################################
# 输入长视频
# 将每个视频裁剪成cut_frame长的片段，多个片段自动编号
##################################################################################


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cut_frame = 30  # 多少帧截一次，自己设置就行
video_sor = r"F:\dataset\2paper_video\9xin_zhengli_saixuan_mp4\lian/2"
video_det = r"F:\dataset\2paper_video\9xin_zhengli_saixuan_mp4\lian/pro"

for root, dirs, files in os.walk(video_sor):  # 这里就填文件夹目录就可以了
    for file in files:
        # 获取文件路径
        if (('.avi' in file) or ("mp4" in file) or (".AVI" in file) or (".MP4" in file) or (".wmv" in file)):
            path = os.path.join(root, file)
            video = cv2.VideoCapture(path)
            video_fps = int(video.get(cv2.CAP_PROP_FPS))
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            totalframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("video_fps:", video_fps)
            current_frame = 1
            file_sp_no = 0
            houzui = file[-4:]
            file_path = video_det
            file_name = file[:-4] + "_" + str(file_sp_no) + ".avi"
            resule = file_path + "/" + file_name
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            out_video = cv2.VideoWriter(resule, fourcc, int(video_fps), (frame_width, frame_height), True)
            # out_video = cv2.VideoWriter(resule, fourcc, 15, (640, 480), True)

            while (True):
                # sleep(1000)
                ret, image = video.read()
                if ret is False:
                    print("没有从", path, "读到帧,可能该文件快结束")
                    video.release()
                    break
                try:
                    image = cv2.resize(image, (frame_width, frame_height))
                except Exception as e:
                    break
                current_frame = current_frame + 1
                if (current_frame <= cut_frame):
                    out_video.write(image)
                    print(current_frame, "/", cut_frame, len(files), "/", len(dirs))
                    # print(" save ")
                if (current_frame > cut_frame):
                    out_video.release()
                    print("-" * 15, "文件", resule, "已保存", "-" * 15)
                    current_frame = 0
                    file_sp_no = file_sp_no + 1
                    # file_path = root.replace(video_sor, video_det)
                    file_path = video_det
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    file_name = file[:-4] + "_" + str(file_sp_no) + ".avi"
                    resule = file_path + "/" + file_name
                    out_video = cv2.VideoWriter(resule, fourcc, int(video_fps), (frame_width, frame_height), True)
                    # out_video = cv2.VideoWriter(resule, fourcc, 15, (640, 480), True)
