import cv2
import numpy as np

# 分别在RGB、红外图片上面找一些位置严格匹配的点
# 第1批融合参数 Tue Mar 15 17:23:11 CST 2022
pts_rgb = np.array([[195, 699], [1679, 762], [826, 341], [1016, 353], [964, 212], [178, 298], [1616, 195], [1177, 90]])
pts_nir = np.array([[247, 687], [1740, 759], [886, 331], [1078, 344], [1034, 202], [243, 284], [1688, 190], [1249, 79]])

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
# pts_rgb = np.array([[399, 159],[109, 940],[1448, 1036], [1106, 870], [935, 583],    [1239, 632],    [1431, 306]])
# pts_nir = np.array([[471, 145],[155, 929],[1512, 1034],[1172, 865],[1007, 575],   [1312, 625],    [1510, 298]])

rgb = cv2.imread(
    r'C:\Users\THHICV\Desktop/ALARM_CCD.jpg')
nir = cv2.imread(
    r'C:\Users\THHICV\Desktop/ALARM_NIR.jpg', 0)

H, W, C = rgb.shape
nirx3 = cv2.cvtColor(nir, cv2.COLOR_GRAY2BGR)  # conver to 3 channel

# 显示出图片，用来确定特征点的坐标
cv2.namedWindow("rgb", 0)
cv2.imshow('rgb', rgb)
cv2.namedWindow("nir", 0)
cv2.imshow('nir', nir)
cv2.waitKey(0)

h1, status = cv2.findHomography(pts_rgb, pts_nir)
# M = cv2.getPerspectiveTransform(pts_rgb,pts_nir)
# print("变换矩阵M为：",M)
# M = [[ 0.997224385234650,-0.00139114429332971,7.74370307364345e-06], [ -0.000462760507708824,0.983599705978519,-1.81922230919884e-05], [-57.1377932431204,17.0913839167372,1]]

duizunrgb = cv2.warpPerspective(rgb, h1, (1920, 1080))

cv2.namedWindow("duizunrgb", 0)
cv2.imshow('duizunrgb', duizunrgb)
cv2.waitKey(0)

# 设置图像的融合的权重
a = 0.5
ronghe = nirx3 * a + duizunrgb * (1 - a)
# ronghe = nirx3 * a + rgb * (1 - a)
# ronghe = cv2.add(nirx3,duizunnir)

ronghe = ronghe.astype(np.uint8)
## 保存图像的融合的结果显示
# cv2.imwrite("ronghe.jpg", ronghe)
cv2.namedWindow("ronghe", 0)
cv2.imshow("ronghe", ronghe)

cv2.waitKey(0)
cv2.destroyAllWindows()
