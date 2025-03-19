from matplotlib import pyplot as plt
from prettytable import PrettyTable
from pycm import *

# C3D
# [[638   0   0]
#  [  1 295   0]
#  [  1   2 127]]

# densnet
# [[637   1   0]
#  [  1 295   0]
#  [  1   2 127]]

# cm = ConfusionMatrix(matrix={0: {0: 638, 1: 0, 2: 0}, 1: {0: 1, 1: 295, 2: 0}, 2: {0: 0, 1: 0, 2: 130}})
cm = ConfusionMatrix(matrix={0: {0: 637, 1: 1, 2: 0}, 1: {0: 1, 1: 295, 2: 0}, 2: {0: 1, 1: 2, 2: 127}})
print(cm)

table = PrettyTable()  # 创建一个表格用于统计每一类的详细信息
table.field_names = ["", "Precision", "Recall", "F1"]
table.add_row(["fire", cm.class_stat['PPV'][0], cm.class_stat['TPR'][0], cm.class_stat['F1'][0]])
table.add_row(["negetive", cm.class_stat['PPV'][1], cm.class_stat['TPR'][1], cm.class_stat['F1'][1]])
table.add_row(["smoke", cm.class_stat['PPV'][2], cm.class_stat['TPR'][2], cm.class_stat['F1'][2]])
print(table)

print(cm.overall_stat['Overall ACC'])
print(cm.overall_stat['PPV Macro'])
print(cm.overall_stat['TPR Macro'])
print(cm.overall_stat['F1 Macro'])
# cm.plot(cmap=plt.cm.Reds,normalized=True,number_label=True,plot_lib="seaborn")
cm.plot(cmap=plt.cm.Blues, number_label=True, plot_lib="matplotlib")

# # coding=utf-8
# import matplotlib.pyplot as plt
# import numpy as np
#
# confusion = np.array( ([ 638,  0,  0],  [ 1,  295,  0],  [ 0,  0,  130]))
# # 热度图，后面是指定的颜色块，可设置其他的不同颜色
# plt.imshow(confusion, cmap=plt.cm.Blues)
# # ticks 坐标轴的坐标点
# # label 坐标轴标签说明
# indices = range(len(confusion))
# # 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
# # plt.xticks(indices, [0, 1, 2])
# # plt.yticks(indices, [0, 1, 2])
# plt.xticks(indices, ['火焰', '负样本', '烟雾'])
# plt.yticks(indices, ['火焰', '负样本', '烟雾'])
#
# plt.colorbar()
#
# plt.xlabel('预测值')
# plt.ylabel('真实值')
# # plt.title('测试混淆矩阵')
#
# # plt.rcParams两行是用于解决标签不能显示汉字的问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 显示数据
# for first_index in range(len(confusion)):  # 第几行
#     for second_index in range(len(confusion[first_index])):  # 第几列
#         # plt.text(first_index, second_index, confusion[first_index][second_index],size = 15, horizontalalignment='center', verticalalignment='center')
#         plt.text(second_index, first_index, confusion[first_index][second_index],size = 15, horizontalalignment='center', verticalalignment='center')
# # 在matlab里面可以对矩阵直接imagesc(confusion)
# # 显示
# plt.show()
