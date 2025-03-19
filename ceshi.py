import os
inRun_dir = r"F:\THHI\program\Fire-Detection-Base-3DCNN\run\20220902-104011_vgg_3D_0.0001_12_THHICV"
rootDir_wandb = os.path.join(inRun_dir, 'wandb')

for dirName in  os.listdir(rootDir_wandb):
    print(os.path.join(rootDir_wandb, str(dirName),"files"))