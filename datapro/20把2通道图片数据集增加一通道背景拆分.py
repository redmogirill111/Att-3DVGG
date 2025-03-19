import os
import cv2
import datetime
import numpy as np

if __name__ == "__main__":
    ###############################################修改此处################################################################
    # 要使用全英文目录，否则cv2.imread读不出来
    # 1、数据集文件组织成以下结构
    # path/fire/ALARM_CCDNIR_20220307-152638_4/00000.png
    # path/negetive/
    # path/smoke/

    rootPath = r"G:\program\date\2paper\2data2_jpg/"
    detRootPath = r"D:\tmp\data2_jpg_cafen/"
    tongDao = 3
    picW = 224
    picH = 224
    ##############################################修改此处#################################################################

    time_process = datetime.datetime.now() - datetime.datetime.now()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    mage = np.zeros((int(picH), int(2 * picW), 3), np.uint8)

    for root, dirs, files in os.walk(rootPath):
        files.sort(key=lambda x: int(x.split(".")[0]))
        for Pic in files:
            if ".jpg" in Pic:
                if "00000.jpg" in Pic:
                    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

                # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
                img = cv2.imread(os.path.join(root, Pic), cv2.IMREAD_COLOR)
                img_mask = bg_subtractor.apply(img)
                # b_rgb, g_rgb, r_rgb = cv2.split(img)
                # chanel_4 = cv2.merge((b_rgb, g_rgb, r_rgb, img_mask))

                # 将当前读取的jpg文件路径+文件名转换为要保存的png文件路径+文件名
                frame_name = os.path.join(root, Pic).replace(os.path.join(rootPath), os.path.join(detRootPath)).replace(
                    ".jpg", ".jpg")

                if not os.path.exists(os.path.dirname(frame_name)):
                    os.makedirs(os.path.dirname(frame_name))
                cv2.imwrite(frame_name, img=img_mask)

            else:
                pass
    cv2.destroyAllWindows()
