import os
import cv2


##########################################
# 将置顶目录下的图片resize到指定尺寸
##########################################

# def resize_img(PicDIR, DetDir, img_size):
def resize_img(PicDIR, img_size):
    w = img_size[0]
    h = img_size[1]

    # 返回path路径下所有文件的名字，以及文件夹的名字，
    for root, dirs, files in os.walk(PicDIR):
        for Pic in files:
            if ".jpg" in Pic:
                # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
                img_array = cv2.imread(os.path.join(root, Pic), cv2.IMREAD_COLOR)
                # 调用cv2.resize函数resize图片
                new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)

                '''生成图片存储的目标路径'''
                save_path = os.path.join(root, Pic)
                cv2.imwrite(save_path, new_array)
            else:
                pass


if __name__ == '__main__':
    # 设置图片路径
    PicDIR = r"F:\dataset\huoyanshujvji11111111\20220313\2-4\YuanMP4-01-Gray/"
    # DetDir = r'D:\tmp\precessvideo/'
    '''设置目标像素大小，此处设为512 * 512'''
    img_size = [171, 128]
    # resize_img(PicDIR, DetDir, img_size)
    resize_img(PicDIR, img_size)

# Resize_Image(r"F:\dataset\huoyanshujvji11111111\20220313\2-4\YuanMP4-01-Gray/")
