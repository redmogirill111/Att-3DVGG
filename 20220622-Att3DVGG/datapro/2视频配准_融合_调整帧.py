import os
import cv2
import numpy as np
import random

# 此处坐标手动查找标志点的坐标、设置融合率
# # 第1批融合参数 Tue Mar 15 17:23:11 CST 2022
# pts_rgb = np.array([[195, 699], [1679, 762], [826, 341], [1016, 353], [964, 212], [178, 298], [1616, 195], [1177, 90]])
# pts_nir = np.array([[247, 687], [1740, 759], [886, 331], [1078, 344], [1034, 202], [243, 284], [1688, 190], [1249, 79]])

# # 第2批融合参数 Tue Mar 15 17:23:11 CST 2022
# pts_rgb = np.array([[447, 718], [463, 330], [733, 343], [949, 319], [1345, 279], [695, 70], [120, 131]])
# pts_nir = np.array([[499, 708], [519, 317], [792, 334], [1009, 308], [1411, 271], [765, 57], [185, 117]])

# # 第3批融合参数 Tue Mar 15 17:23:11 CST 2022
# pts_rgb = np.array([[997, 658], [264, 511], [338, 77], [1187, 366], [1031, 563], [1506, 388], [1656, 394]])
# pts_nir = np.array([[1055, 648], [320, 498], [398, 64], [1256, 357], [1089, 555], [1575, 382], [1728, 388]])

# # 第4批融合参数 Tue Mar 15 17:23:11 CST 2022
# pts_rgb = np.array([[263, 795],[443, 632],[770, 149], [826, 794], [1158, 535],    [1431, 383],    [1280, 835]])
# pts_nir = np.array([[325, 786],[507, 622],[839, 137],[891, 786],[1228, 527],   [1503, 375],    [1348, 831]])

# 第5批融合参数 Tue Mar 15 17:23:11 CST 2022
pts_rgb = np.array([[399, 159], [109, 940], [1448, 1036], [1106, 870], [935, 583], [1239, 632], [1431, 306]])
pts_nir = np.array([[471, 145], [155, 929], [1512, 1034], [1172, 865], [1007, 575], [1312, 625], [1510, 298]])

ronghelv = 0.5

mp4_rgb = r"F:\dataset\红外和可见光的火焰数据集\2022年3月13日一期采集\1-1配准融合\第五次配准/CCD/"
mp4_nir = r"F:\dataset\红外和可见光的火焰数据集\2022年3月13日一期采集\1-1配准融合\第五次配准/NIR/"
mp4_des = r"F:\dataset\红外和可见光的火焰数据集\2022年3月13日一期采集\2-1配准原视频不融合/第五次配准/"

# mp4_rgb = r"/home/thhicv/视频/定位/CCD/"
# mp4_nir = r"/home/thhicv/视频/定位/NIR/"
# mp4_des = "/home/thhicv/视频/定位/CCD-NIR/"

# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if not os.path.exists(mp4_des):
    os.makedirs(mp4_des)
for root, dirs, files in os.walk(mp4_rgb):
    j = 1
    for file in files:
        print("the file '", file, "' is read ")
        # 获取文件路径
        v_rgb = cv2.VideoCapture(os.path.join(mp4_rgb, file))
        v_nir = cv2.VideoCapture(os.path.join(mp4_nir, file.replace("CCD", "NIR")))

        if v_rgb.isOpened() == False or v_nir.isOpened() == False:
            print("*" * 20, "waring!", "*" * 20, "file not found!")
            continue
        print("\nv_rgb path is :", os.path.join(mp4_rgb, file), "  The v_rgb video is opend:", v_rgb.isOpened())
        print("v_nir path is :", os.path.join(mp4_nir, file.replace("CCD", "NIR")), "  The v_nir video is opend:",
              v_nir.isOpened())

        # 获取第一个v_rgb视频的参数，保证红外和可见光视频fps、分辨率一致。程序自动适应不同总的帧数进行合并
        fps = v_rgb.get(cv2.CAP_PROP_FPS)
        w = v_rgb.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = v_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # fps = 15
        # w = 1920
        # h = 1080

        totalframes_ccd = int(v_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        totalframes_nir = int(v_nir.get(cv2.CAP_PROP_FRAME_COUNT))
        totalframes = int(min(v_rgb.get(cv2.CAP_PROP_FRAME_COUNT), v_nir.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_dist = (totalframes_nir - totalframes_ccd + 5) if (totalframes_nir > totalframes_ccd) else (
                totalframes_ccd - totalframes_nir + 8)
        print('The v_rgb video w: {}, h: {}, totalframes_ccd: {}, fps: {}，frame_dist:{}帧'.format(w, h, totalframes_ccd,
                                                                                                 fps, frame_dist))

        # 准备保存视频的参数 如果保存的是mage 宽度设置为3*w 如果保存的是处理后的图片 宽度设置为w
        # cv2.namedWindow("windows", 0)
        out_ronghe = cv2.VideoWriter(os.path.join(mp4_des, file.replace("CCD", "CCD-NIR")), fourcc, fps,
                                     (int(w), int(h)), True)
        out_mage = cv2.VideoWriter(os.path.join(mp4_des, file.replace("CCD", "CCDNIR")), fourcc, fps,
                                   (int(2 * w), int(h)), True)
        out_nir = cv2.VideoWriter(os.path.join(mp4_des, file.replace("CCD", "NIR-PRO")), fourcc, fps,
                                  (int(w), int(h)), True)
        out_rgb = cv2.VideoWriter(os.path.join(mp4_des, file.replace("CCD", "CCD-PRO")), fourcc, fps,
                                  (int(w), int(h)), True)
        mage = np.zeros((int(h), int(2 * w), 3), np.uint8)
        m, status = cv2.findHomography(pts_nir, pts_rgb)

        i = 0

        while (totalframes - i) >= 0:
            # # 此处代码为了约束红外和可见光的帧差
            # if (totalframes_nir < totalframes_ccd) and (frame_dist>0):
            # 	print("跳过NIR开头第", frame_dist, "帧")
            # 	frame_dist -= 1
            # 	rval_nir, frame_nir = v_nir.read()
            # 	continue
            # elif (totalframes_nir > totalframes_ccd) and (frame_dist>0):
            # 	print("跳过CCD开头第", frame_dist, "帧")
            # 	frame_dist -= 1
            # 	rval_rgb, frame_rgb = v_rgb.read()
            # 	continue
            rval_rgb, frame_rgb = v_rgb.read()
            rval_nir, frame_nir = v_nir.read()
            if rval_nir == False or rval_rgb == False:
                print("frame not found!,continue!  ", random.randint(0, 9999999))
                i += 1
                continue
            # print("rval_rgb is :", rval_rgb, "    frame_rgb is ", frame_rgb[0, 0], "    frame_rgb.shape is ",
            #       frame_rgb.shape)
            # print("rval_nir is :", rval_nir, "    frame_nir is ", frame_nir[0, 0], "    frame_nir.shape is ",
            #       frame_nir.shape)

            # # 将红外单通道转换成三通道
            # frame_nir = cv2.cvtColor(frame_nir, cv2.COLOR_GRAY2BGR)
            frame_nir_duizun = cv2.warpPerspective(frame_nir, m, (1920, 1080))

            # # 将红外单通道转换为伪彩色
            # frame_nir_duizun = cv2.applyColorMap(frame_nir_duizun, cv2.COLORMAP_JET)

            # # 将红外单通道转换为红色
            # frame_nir_duizun[:, :, 0] = 0
            # frame_nir_duizun[:, :, 1] = 0

            frame_ronghe = frame_rgb * ronghelv + frame_nir_duizun * (1 - ronghelv)
            # frame_ronghe = cv2.add(frame_rgb, frame_nir_duizun)

            # # 判断RBG的帧和融合后的帧是否相同，相同则说明红外帧为空
            # difference = cv2.subtract(frame_ronghe, frame_rgb)
            # result = not np.any(difference)
            # if result is True:
            #     print("两张图片一样")
            # else:
            #     print("两张图片不一样")

            frame_ronghe = frame_ronghe.astype(np.uint8)

            mage[0:int(h), 0:int(w)] = frame_rgb
            mage[0:int(h), int(w):int(2 * w)] = frame_nir_duizun
            # mage[0:int(h), int(2 * w):int(3 * w)] = frame_ronghe

            # 修改此处切换显示或者保存的图像 frame_ronghe/mage
            out_ronghe.write(frame_ronghe)
            out_mage.write(mage)
            out_nir.write(frame_nir_duizun)
            out_rgb.write(frame_rgb)
            # cv2.imshow("windows", mage)
            i += 1
            print("总进度：", j, "/", len(files), "   当前文件进度:", str(i), "/", totalframes)
        print("the file saved in ", os.path.join(mp4_des, file.replace("CCD", "CCD-NIR")))
        # out_file.release()
        j += 1
        print("-----------------------------------------------------------" * 2)
