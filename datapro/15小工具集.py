# 功能：对推理后的结果进行整理
# import os
# import pandas as pd
#
# data = pd.read_table(r"D:\tmp\video_dataset_test\3result_NewPro.txt", sep=" ", header=None)
# data.sort_values(by=0)
# det_file = r"F:\dataset\2paper_video\3fenlei\1/"
# name_old = ""
# thr_max = 0.01
# thr_old = 0.01
# name_max = ""
# table_max = ""
# # print(data.shape)
# first = 1
#
# for id, raw in data.iterrows():
#     name_now = raw[0]
#     table_now = raw[1]
#     thr_now = raw[2]
#     if first == 1:
#         thr_max = thr_now
#         name_max = name_now
#         table_max = table_now
#         thr_old = thr_now
#         name_old = name_now
#         table_old = table_now
#         first = 0
#     if name_now == name_old:
#         if thr_now > thr_max:
#             thr_max = thr_now
#             name_max = name_now
#             table_max = table_now
#             thr_old = thr_now
#             name_old = name_now
#             table_old = table_now
#         else:
#             pass
#     else:
#         label_file = os.path.join(det_file, table_max, "")
#         print("mv " + name_max + " " + label_file)
#         thr_old = thr_now
#         name_old = name_now
#         table_old = table_now
#         thr_max = thr_now
#         name_max = name_now
#         table_max = table_now


# # 功能：从metadata_02242020.json文件中解析出下载链接保存为metadata_02242020.txt
# # 场景：方便筛选和调用下载工具下载文件
# import pandas as pd
# frame = pd.read_json('metadata_02242020.json')
# for id, row in frame.iterrows():
#     if row["label_state"] == 23:
#         url_part = row["url_part"]
#         url_root = row["url_root"]
#         file_name = row["file_name"]
#         # url = "negetive\t" + url_root + url_part+"\n"
#         url = "negetive\t" + file_name+".mp4\n"
#
#         tfile = open('metadata_02242020.txt', 'a')
#         tfile.write(url)
#         tfile.close()
#     if row["label_state"] == 16:
#         url_part = row["url_part"]
#         url_root = row["url_root"]
#         file_name = row["file_name"]
#         url = "smoke\t" + file_name+".mp4\n"
#         # url = "smoke\t" + url_root + url_part+"\n"
#
#         tfile = open('metadata_02242020.txt', 'a')
#         tfile.write(url)
#         tfile.close()
