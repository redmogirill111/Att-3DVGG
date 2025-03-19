import os

number = 0

for root, dirs, files in os.walk(r"F:\dataset\2paper_video\9xin_zhengli_saixuan_mp4\fin_dataset\dataset_jpg"):
    # if len(files) <= 17 and "ALARM" in root:
    if len(files) <= 17:
        # print(root," ", len(files))
        # os.removedirs(root)
        print(root, "\t", len(files))
print("*" * 80)
import os

train_fire = 0
train_smoke = 0
train_neg = 0
test_fire = 0
test_smoke = 0
test_neg = 0
val_fire = 0
val_smoke = 0
val_neg = 0

for root, dirs, files in os.walk(r"F:\dataset\2paper_video\9xin_zhengli_saixuan_mp4\fin_dataset\dataset_jpg"):
    if "train" in root and "fire" in root:
        train_fire = train_fire + 1
    if "train" in root and "smoke" in root:
        train_smoke = train_smoke + 1
    if "train" in root and "negetive" in root:
        train_neg = train_neg + 1
    if "test" in root and "fire" in root:
        test_fire = test_fire + 1
    if "test" in root and "smoke" in root:
        test_smoke = test_smoke + 1
    if "test" in root and "negetive" in root:
        test_neg = test_neg + 1
    if "val" in root and "fire" in root:
        val_fire = val_fire + 1
    if "val" in root and "smoke" in root:
        val_smoke = val_smoke + 1
    if "val" in root and "negetive" in root:
        val_neg = val_neg + 1
    # print(root, " ", len(files))
print("\ntrain_fire", "\t", "train_neg", "\t", "train_smoke", "\t", "test_fire", "\t", "test_neg", "\t", "test_smoke",
      "\t", "val_fire", "\t", "val_smoke", "\t", "val_neg")
print(train_fire, "\t", "\t", train_neg, "\t", "\t", train_smoke, "\t", "\t", "\t", test_fire, "\t", "\t", test_neg,
      "\t", "\t", test_smoke, "\t", "\t", val_fire, "\t", "\t", val_smoke, "\t", "\t", val_neg)
