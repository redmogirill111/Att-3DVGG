import timeit
from datetime import datetime
import socket
import os
from os import listdir
import glob

import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
import shutil
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, T2CC3D_model, DIY, C3D_AttNAtt, p3d_model, R2Plus1D_atten, \
    resnet_3d, xception_3D, mobilenetv2, shufflenetv2, squeezenet, MobileNetV3, VGG3D_AttNatt, DIY_X, F_6M3DC, Dahan_3D, \
    DenseNet_3D

# 此代码基于https://github.com/jfzhang95/pytorch-video-recognition重构，用于烟火识别
# 作者：宋俊猛 邮箱：1@vn.mk

nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 100  # Store a model every snapshot epochs
lr = 1e-3  # Learning rate
weight_decay = 1e-4
lr_list = []
dataset = 'ucf101'  # Options: hmdb51 or ucf101
modelName = 'Dahan_3D'
# 待测试 Xception mobilenetv2 ShuffleNetV2 SqueezeNet MobileNetV3
# C3D R2Plus1D R3D VGG DIY R2Plus1D vgg_3D t2CC3D C3D_AttNAtt P3D199 resnet10 Xception VGG3D_Att F_6M3DC Dahan_3D DenseNet_3D
saveName = modelName + '-' + dataset
acc_max = 0.0

# 适配两台不同机器调用
select_device = "Q"

if select_device == "Q":
    num_worker = 2
    bz = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif select_device == "V":
    num_worker = 4
    bz = 20
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
else:
    num_worker = 1
    bz = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device being used:", device)
if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 3
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(save_dir_root):
    os.mkdir(save_dir_root)
print("保存路径是：", save_dir_root)
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
print(save_dir)


def confusion_matrix_calc_pr(labels_truth, labels_predict, class_name=[]):
    # labels_truth is ndarray like [0. 0. 1.]
    # labels_predict is ndarray
    # class_name is list like [0, 1]

    sum_tp = 0
    matrix = confusion_matrix(labels_truth, labels_predict)
    if len(class_name) == 0:
        labels_list = np.arange(0, matrix.shape[1]).tolist()
    else:
        labels_list = class_name

    n = np.sum(matrix)
    for i in range(len(labels_list)):
        sum_tp += matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
    acc = sum_tp / n  # 总体准确率
    # print("the model accuracy is ", acc)

    # kappa
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col

    # precision, recall, specificity
    table = PrettyTable()  # 创建一个表格
    table.field_names = ["", "Precision", "Recall", "F1"]
    Precision_all = 0.0
    Recall_all = 0.0
    F1_all = 0.0
    for i in range(len(labels_list)):  # 精确度、召回率、特异度的计算
        TP = matrix[i, i]
        FP = np.sum(matrix[i, :]) - TP
        FN = np.sum(matrix[:, i]) - TP
        TN = np.sum(matrix) - TP - FP - FN

        Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
        Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
        F1 = round(2.0 * Precision * Recall / (Precision + Recall), 2) if TP + FP != 0 else 0.
        # Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

        # table.add_row([labels[i], Precision, Recall, Specificity])
        table.add_row([labels_list[i], Precision, Recall, F1])
        Precision_all = Precision + Precision_all
        Recall_all = Recall + Recall_all
        F1_all = F1 + F1_all
    print(table)
    return acc, Precision_all / 3.0, Recall_all / 3.0, F1_all / 3.0, matrix, table


def plot_cm(confusion, save_path, epoch, class_name=[]):  # numname of svg
    if not os.path.exists(save_path):
        os.makedirs(str(save_path))
    fig = plt.figure(dpi=300, figsize=(10, 10))  ##dpi调整图片的DPI大小，figsize调整画布大小
    # plt.imshow(confusion, cmap='binary', interpolation='nearest')  # 画混淆矩阵的函数,数量的颜色
    # 就是坐标轴含义说明了

    np.set_printoptions(precision=2)  # 确定小数点后的位数，也就是精度
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]  # np.newaxis插入新维度，变成一个列向量
    # print(confusion_normalized)#显示混淆矩阵数字

    plt.imshow(confusion_normalized, cmap='Blues', interpolation='nearest')  # 画混淆矩阵的函数，比例的颜色blues
    ##plt.imshow(confusion_normalized, cmap='Binary', interpolation='nearest')  # 画混淆矩阵的函数，比例的颜色lack

    ## ytick = xtick = np.arange(0, 10)
    # Lable_list = ['Diving', 'Golf Swing', 'Kicking', 'Lifting', 'Riding Horse', 'Running', 'Skateboarding',
    #               'Swing-Bench', 'Swing-Side', 'Walking']

    if len(class_name) == 0:
        Lable_list = np.arange(0, confusion.shape[1]).tolist()
    else:
        Lable_list = class_name

    plt.xticks(range(len(confusion_normalized)), Lable_list, fontsize=18, rotation=45)  # rotation表示倾斜的程度
    plt.yticks(range(len(confusion_normalized)), Lable_list, fontsize=18)

    # 显示数据，直观些 2是一个阈值
    thresh = confusion_normalized.max() / 2
    for i in range(len(confusion_normalized)):
        for j in range(len(confusion_normalized[i])):
            # color混淆矩阵中的数字颜色，fontsize表示字体大小，va,ha表示位置
            plt.text(i, j, round(confusion_normalized[j, i], 2),
                     color='white' if confusion_normalized[i][j] > thresh else "black", va='center', ha='center',
                     fontsize=10)  ##调整画出来的图像为白黑色
    plt.ion()  # 显示
    # plt.pause(2)  # 暂停2s 取消显示图
    fig.savefig(save_path + 'epoch-' + str(epoch) + '.svg', format='svg')  # 保存图像为svg，前面为地址，后面为类型
    plt.savefig(save_path + 'epoch-' + str(epoch) + '.svg', format='svg')  # 保存图像为svg，前面为地址，后面为类型
    plt.close('all')  # 关闭显示窗口


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            dataset:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == 'VGG':
        model = T2CC3D_model.vgg_3D(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'DIY':
        model = DIY.DIY(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_atten.R2Plus1D(num_classes=3, layer_sizes=(3, 4, 6, 3))
        train_params = model.parameters()
    elif modelName == 'vgg_3D':
        model = T2CC3D_model.vgg_3D(num_classes=3)
        train_params = model.parameters()
    elif modelName == 't2CC3D':
        model = T2CC3D_model.t2CC3D(num_classes=3)
        train_params = model.parameters()
    elif modelName == 'C3D_AttNAtt':
        model = C3D_AttNAtt.C3D_AttNAtt2(sample_size=112, num_classes=3,
                                         lstm_hidden_size=512, lstm_num_layers=3)
        train_params = model.parameters()
    elif modelName == 'P3D199':
        model = p3d_model.P3D199(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'resnet10':
        model = resnet_3d.resnet10(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'Xception':
        model = xception_3D.Xception(num_classes=num_classes)
        train_params = model.parameters()
    elif modelName == 'mobilenetv2':
        model = mobilenetv2.get_model(num_classes=3, sample_size=112, width_mult=1.)
        train_params = model.parameters()
    elif modelName == 'ShuffleNetV2':
        model = shufflenetv2.get_model(num_classes=3, sample_size=112, width_mult=1.)
        train_params = model.parameters()
    elif modelName == 'SqueezeNet':
        model = squeezenet.get_model(num_classes=3, sample_size=112, width_mult=1.)
        train_params = model.parameters()
    elif modelName == 'mobilenetv3_large':
        model = MobileNetV3.get_model(num_classes=3, sample_size=112, width_mult=1.)
        train_params = model.parameters()
    elif modelName == 'DIY_X':
        model = DIY_X.Xception(num_classes=3)
        train_params = model.parameters()
    elif modelName == 'VGG3D_Att':
        model = VGG3D_AttNatt.VGG3D_Att(num_classes=3)
        train_params = model.parameters()
    elif modelName == 'F_6M3DC':
        model = F_6M3DC.F_6M3DC(num_classes=3)
        train_params = model.parameters()
    elif modelName == 'Dahan_3D':
        model = Dahan_3D.Dahan_3D(num_classes=3)
        train_params = model.parameters()
    elif modelName == 'DenseNet_3D':
        model = DenseNet_3D.DenseNet(num_init_features=64,
                                     growth_rate=32,
                                     # block_config=(6, 12, 24, 16))
                                     block_config=(32, 64, 128))
        train_params = model.parameters()

    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification

    # optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # 2022年3月30日 已测试不收敛

    # 测试可以收敛 Loss下降慢，抖动严重
    # optimizer = optim.Adam(train_params, lr=lr, weight_decay=0.0001)

    optimizer = optim.Adam(train_params, lr=lr, weight_decay=weight_decay)

    # optimizer = torch.optim.Adadelta(train_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

    # optimizer = torch.optim.Adagrad(train_params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)

    # optimizer = torch.optim.RMSprop(train_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    # optimizer = torch.optim.SGD(train_params, lr=0.02, momentum=0.95)

    # optimizer = torch.optim.SGD(train_params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4,
                                          gamma=0.95)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
    #                                       gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # https://github.com/mengcius/PyTorch-Learning-Rate-Scheduler/blob/master/PyTorch%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%B0%83%E6%95%B4%E7%AD%96%E7%95%A5.ipynb
    # torch.optim.lr_scheduler.LambdaLR
    # 自定义lamda函数
    # torch.optim.lr_scheduler.StepLR
    # 等间隔阶梯下降
    # torch.optim.lr_scheduler.MultiStepLR
    # 指定多间隔step_list阶梯下降
    # torch.optim.lr_scheduler.ExponentialLR
    # 指数下降
    # torch.optim.lr_scheduler.CosineAnnealingLR
    # 余弦退火
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    # 带热启动的余弦退火
    # torch.optim.lr_scheduler.CyclicLR
    # 循环调整
    # torch.optim.lr_scheduler.OneCycleLR
    # 第一次退火到大学习率
    # torch.optim.lr_scheduler.ReduceLROnPlateau
    # 自适应下降

    if resume_epoch == 0:
        # print("Training {} from scratch...".format(modelName))
        print("从头开始训练 {} 模型...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=log_dir)
    log_txt = save_dir + "/" + "train_log.txt"
    with open(log_txt, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        pass

    print('Training model on {} dataset...'.format(dataset))

    shutil.copy('dataloaders/dataset.py', save_dir)
    # shutil.copy('network/C3D_model.py', log_dir)
    # shutil.copy('network/R2Plus1D_model.py', log_dir)
    # shutil.copy('network/R3D_model.py', log_dir)
    # shutil.copy('network/DIY.py', log_dir)
    shutil.copy('train.py', save_dir)
    shutil.copy('mypath.py', save_dir)
    shutil.copytree('network', os.path.join(save_dir, "network"))

    # resize_height = {int}128
    # resize_width = {int}171
    # 将图片裁剪到crop_size*crop_size的尺寸
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=bz,
                                  shuffle=True, num_workers=num_worker)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=bz,
                                num_workers=num_worker)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=bz,
                                 num_workers=num_worker)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    print("trainval_sizes:", str(trainval_sizes), "   test_size:", str(test_size))

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.

            LABEL_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的真实值
            PRED_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的预测值

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()

                # 2022年3月30日11:20:46 设置学习率衰减策略
                # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

                # 2022年3月30日11:22:50 阶梯下降
                lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

                model.train()
            else:
                model.eval()

            # torch.Size([20, 3, 16, 112, 112])
            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs).to(device)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.long())

                LABEL_TrainVal = np.concatenate((LABEL_TrainVal, labels.cpu().numpy()), -1)
                PRED_TrainVal = np.concatenate((PRED_TrainVal, preds.cpu().numpy()), -1)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            # LABEL_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的真实值
            # PRED_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的预测值

            # LABEL_TrainVal = np.concatenate((LABEL_TrainVal, labels.cpu().numpy()), -1)
            # PRED_TrainVal = np.concatenate((PRED_TrainVal, preds.cpu().numpy()), -1)

            Accuracy_TrainVal, Precision_TrainVal, Recall_TrainVal, F1_TrainVal, confusion_matrix_TrainVal, tabel_TrainVal = confusion_matrix_calc_pr(
                LABEL_TrainVal, PRED_TrainVal, class_name=["fire", "negetive", "smoke"])
            plot_cm(confusion_matrix_TrainVal, os.path.join(save_dir, 'confusion_matrix', 'TrainVal'), str(epoch),
                    class_name=["fire", "negetive", "smoke"])
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', Accuracy_TrainVal, epoch)
                writer.add_scalar('Train/Precision', Precision_TrainVal, epoch)
                writer.add_scalar('Train/Recall', Recall_TrainVal, epoch)
                writer.add_scalar('Train/F1', F1_TrainVal, epoch)
                writer.add_pr_curve('Train/Fire', Precision_TrainVal, Recall_TrainVal, epoch)

            else:
                writer.add_scalar('Val/Loss', epoch_loss, epoch)
                writer.add_scalar('Val/Accuracy', Accuracy_TrainVal, epoch)
                writer.add_scalar('Val/Precision', Precision_TrainVal, epoch)
                writer.add_scalar('Val/Recall', Recall_TrainVal, epoch)
                writer.add_scalar('Val/F1', F1_TrainVal, epoch)
                writer.add_pr_curve('Val/Fire', Precision_TrainVal, Recall_TrainVal, epoch)

            print("\n[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            print("[CM] Accuracy_TrainVal:{} Precision_TrainVal:{} Recall_TrainVal:{} F1_TrainVal:{}".format(
                Accuracy_TrainVal, Precision_TrainVal, Recall_TrainVal, F1_TrainVal))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))

        if epoch % save_epoch == (save_epoch - 1):
            # torch.save({
            #     'epoch': epoch + 1,
            #     'state_dict': model.state_dict(),
            #     'opt_dict': optimizer.state_dict(),
            # }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            LABEL = np.array([])  ##定义一个数组存放混淆矩阵中的真实值
            PRED = np.array([])  ##定义一个数组存放混淆矩阵中的预测值
            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data.long())

                # # 添加输出混淆矩阵
                LABEL = np.concatenate((LABEL, labels.cpu().numpy()), -1)
                PRED = np.concatenate((PRED, preds.cpu().numpy()), -1)

            Accuracy_test, Precision_test, Recall_test, F1_test, confusion_matrix_test, tabel_test = confusion_matrix_calc_pr(
                LABEL,
                PRED,
                class_name=[
                    "fire",
                    "negetive",
                    "smoke"])
            timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
            with open(log_txt, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                f.write(
                    str(timestamp) + "\t" + str(epoch) + "\t" + "Accuracy_test:" + str(
                        Accuracy_test * 100) + "\t" + "Precision_test:" + str(
                        Precision_test * 100) + "\t" + "Recall_test:" + str(
                        Recall_test * 100) + "\t" + "F1_test:" + str(F1_test * 100) + "\n" + str(tabel_test) + "\n")
            global acc_max
            if Accuracy_test > acc_max:
                acc_max = Accuracy_test
                print("Max Accuracy is :", acc_max, '\n')
                writer.add_scalar('Max/Accuracy', Accuracy_test, epoch)
                writer.add_scalar('Max/Precision', Precision_test, epoch)
                writer.add_scalar('Max/Recall', Recall_test, epoch)
                writer.add_scalar('Max/F1', F1_test, epoch)

                timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
                with open(log_txt, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                    f.write(
                        str(timestamp) + "\t" + str(epoch) + "\t" + "MAX_Accuracy_test:" + str(
                            Accuracy_test * 100) + "\t" + "MAX_Precision_test:" + str(
                            Precision_test * 100) + "\t" + "MAX_Recall_test:" + str(
                            Recall_test * 100) + "\t" + "MAX_F1_test:" + str(F1_test * 100) + "\n")
                # 删除已保存的所有模型，再保存一个最优的模型
                for file_name in listdir(os.path.join(save_dir, 'models')):
                    if file_name.endswith('.pth.tar'):
                        os.remove(os.path.join(save_dir, 'models', file_name))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models',
                                saveName + '_epoch-' + str(epoch) + '_acc-' + str(acc_max)[:6] + '.pth.tar'))
                print("Save model at {}\n".format(
                    os.path.join(save_dir, 'models',
                                 saveName + '_epoch-' + str(epoch) + '_acc-' + str(acc_max)[:6] + '.pth.tar')))

            plot_cm(confusion_matrix_test, os.path.join(save_dir, 'confusion_matrix', 'Test'), str(epoch),
                    class_name=["fire", "negetive", "smoke"])

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('Test/Loss', epoch_loss, epoch)
            # writer.add_scalar('Test/test', epoch_acc, epoch)

            writer.add_scalar('Test/Accuracy', Accuracy_test, epoch)
            writer.add_scalar('Test/Precision', Precision_test, epoch)
            writer.add_scalar('Test/Recall', Recall_test, epoch)
            writer.add_scalar('Test/F1', F1_test, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            print("[CM] Accuracy_test:{} Precision_test:{} Recall_test:{} F1_test:{}".format(Accuracy_test,
                                                                                             Precision_test,
                                                                                             Recall_test, F1_test))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    # # 画出网络结构图
    # # model.to("cpu")
    # WC3D = SummaryWriter(comment=modelName, log_dir=log_dir)
    # WC3D.add_graph(model, inputs, True)
    # WC3D.close()
    writer.close()


if __name__ == "__main__":
    train_model()
