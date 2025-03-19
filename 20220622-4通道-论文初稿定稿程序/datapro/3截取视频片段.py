import os
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
cut_frame = 90  # 多少帧截一次，自己设置就行
video_sor = r"F:\dataset\红外和可见光的火焰数据集\2022年3月13日一期采集\2-2分割时间\2-1 视频\smoke/"
video_det = r"F:\dataset\红外和可见光的火焰数据集\2022年3月13日一期采集\2-2分割时间\2-1 视频\2smoke/"

for root, dirs, files in os.walk(video_sor):  # 这里就填文件夹目录就可以了
    for file in files:
        # 获取文件路径
        if ('.avi' in file):
            path = os.path.join(root, file)
            video = cv2.VideoCapture(path)
            video_fps = int(video.get(cv2.CAP_PROP_FPS))
            print("video_fps:", video_fps)
            current_frame = 1
            file_sp_no = 0
            # file_path = root.replace(video_sor, video_det)
            file_path = video_det
            file_name = file[:-4] + "_" + str(file_sp_no) + '.avi'
            resule = file_path + "/" + file_name
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(w, h)
            out_video = cv2.VideoWriter(resule, fourcc, int(video_fps), (3840, 1080), True)

            while (True):
                ret, image = video.read()
                current_frame = current_frame + 1
                if ret is False:
                    print("没有从", path, "读到帧,可能该文件快结束")
                    video.release()
                    break
                if (current_frame < cut_frame):
                    out_video.write(image)
                    print(current_frame, "/", cut_frame, len(files), "/", len(dirs))
                    # print(" save ")
                if (current_frame >= cut_frame):
                    out_video.release()
                    print("-" * 15, "文件", resule, "已保存", "-" * 15)
                    current_frame = 0
                    file_sp_no = file_sp_no + 1
                    # file_path = root.replace(video_sor, video_det)
                    file_path = video_det
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    file_name = file[:-4] + "_" + str(file_sp_no) + '.avi'
                    resule = file_path + "/" + file_name
                    out_video = cv2.VideoWriter(resule, fourcc, int(video_fps), (3840, 1080), True)
