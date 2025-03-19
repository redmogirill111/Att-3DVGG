import os
number = 0

for root, dirs, files in os.walk(r"F:\dataset\huoyanshujvji11111111\20220313\2-4\RongHe"):
    if len(files)  <= 17 and "ALARM"in root:
        # print(root," ", len(files))
        # os.removedirs(root)
        print(root)
        print("*"*40)
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

for root, dirs, files in os.walk(r"F:\dataset\huoyanshujvji11111111\20220313\2-4\RongHe"):
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
    print(root," ",len(files))

print(train_fire,train_neg,train_smoke,test_fire,test_neg,test_fire,val_fire,val_smoke,val_neg)
