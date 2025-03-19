import time
from datetime import datetime
import socket
import os
from os import listdir
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from prettytable import PrettyTable
from pycm import *
import wandb
# wandb sync wandb/dryrun-folder-name 在线提交记录
# https://wandb.ai/fusang/ 查看训练记录
# pip install wandb
# Run `wandb offline` to turn off syncing.
# wandb login
# cef070a0c15100d6b4a4678afd3e5c0ebc7d5062
import config
from torchvision import transforms
from dataloaders.dataset import DatasetFromLMDB
import shutil
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, T2CC3D_model, DIY, C3D_AttNAtt, p3d_model, R2Plus1D_atten, \
    resnet_3d, xception_3D, mobilenetv2, shufflenetv2, squeezenet, MobileNetV3, VGG3D_AttNatt, DIY_X, F_6M3DC, Dahan_3D, \
    DenseNet_3D, GoogLeNet, vgg11_3d, slowfastnet_small
from network.TSM_Att import models as TSN_model
from network.TSM_Att.spatial_transforms import *
from network.TSM.models import TSN
from network.TSM.transforms import *

def confusion_matrix_calc_pr(labels_truth, labels_predict):
    # 更新此函数(2022年7月25日)，使用第三方库 https://github.com/sepandhaghighi/pycm 重新实现
    # 参考文献 https://blog.csdn.net/weixin_45825073/article/details/122042165
    # pycm官方手册 https://www.pycm.io/doc/index.html
    cm_new = ConfusionMatrix(actual_vector=labels_truth, predict_vector=labels_predict)
    matrix = cm_new.matrix

    # 输出个类别的precision, recall, f1等指标
    table = PrettyTable()  # 创建一个表格用于统计每一类的详细信息
    table.field_names = ["", "Precision", "Recall", "F1"]
    table.add_row(["fire", cm_new.class_stat['PPV'][0], cm_new.class_stat['TPR'][0], cm_new.class_stat['F1'][0]])
    table.add_row(["negetive", cm_new.class_stat['PPV'][1], cm_new.class_stat['TPR'][1], cm_new.class_stat['F1'][1]])
    table.add_row(["smoke", cm_new.class_stat['PPV'][2], cm_new.class_stat['TPR'][2], cm_new.class_stat['F1'][2]])
    # print(table)

    return cm_new.overall_stat['Overall ACC'] if cm_new.overall_stat['Overall ACC'] != "None" else 0, \
           cm_new.overall_stat['PPV Macro'] if cm_new.overall_stat['PPV Macro'] != "None" else 0, \
           cm_new.overall_stat['TPR Macro'] if cm_new.overall_stat['TPR Macro'] != "None" else 0, \
           cm_new.overall_stat['F1 Macro'] if cm_new.overall_stat['F1 Macro'] != "None" else 0, \
           matrix, table

    ## 若需要计算其他参数，使用以下代码即可
    # from pycm import *
    # cm = ConfusionMatrix(matrix={0: {0: 626, 1: 7, 2: 5}, 1: {0: 2, 1: 293, 2: 1}, 2: {0: 3, 1: 1, 2: 126}})
    # print(cm)
    # 原函数删除2022年8月9日


def main(inRun_dir, log_txt, resume_epoch, resume_pth, epochAll, test_interval, useTest):
    testAcc_max = 0.0
    valAcc_max = 0.0

    for epoch in range(resume_epoch, epochAll):
        timeEpochStart = time.clock()
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            timeTrainvalStart = time.clock()
            # reset the running loss and corrects
            running_loss = 0.00
            running_corrects = 0.00
            LABEL_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的真实值
            PRED_TrainVal = np.array([])  ##定义一个数组存放混淆矩阵中的预测值

            if phase == 'train':
                model.train()
            else:
                model.eval()
            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs).to(device)
                else:
                    with torch.no_grad():
                        outputs = model(inputs).to(device)

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

            Accuracy_TrainVal, Precision_TrainVal, Recall_TrainVal, F1_TrainVal, confusion_matrix_TrainVal, tabel_TrainVal = confusion_matrix_calc_pr(
                LABEL_TrainVal, PRED_TrainVal)

            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train/Accuracy', Accuracy_TrainVal, epoch)
                writer.add_scalar('Train/Precision', Precision_TrainVal, epoch)
                writer.add_scalar('Train/Recall', Recall_TrainVal, epoch)
                writer.add_scalar('Train/F1', F1_TrainVal, epoch)
                wandb.log({'Train/Accuracy': Accuracy_TrainVal,
                           'Train/F1': F1_TrainVal,
                           'Train/Precision': Precision_TrainVal,
                           'Train/Recall': Recall_TrainVal,
                           'Train/Loss': epoch_loss, 'Train/epoch': epoch,
                           'Train/LR': optimizer.param_groups[0]['lr'],
                           })

            else:
                writer.add_scalar('Val/Loss', epoch_loss, epoch)
                writer.add_scalar('Val/LR', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Val/Accuracy', Accuracy_TrainVal, epoch)
                writer.add_scalar('Val/Precision', Precision_TrainVal, epoch)
                writer.add_scalar('Val/Recall', Recall_TrainVal, epoch)
                writer.add_scalar('Val/F1', F1_TrainVal, epoch)
                wandb.log({'Val/Accuracy': Accuracy_TrainVal,
                           'Val/F1': F1_TrainVal,
                           'Val/Precision': Precision_TrainVal,
                           'Val/Recall': Recall_TrainVal,
                           'Val/Loss': epoch_loss, 'Val/epoch': epoch,
                           'Val/LR': optimizer.param_groups[0]['lr'],
                           })

                if Accuracy_TrainVal > valAcc_max:
                    valAcc_max = Accuracy_TrainVal
                    print("MaxVal Accuracy is :", valAcc_max, '\n')
                    writer.add_scalar('MaxVal/Accuracy', valAcc_max, epoch)
                    writer.add_scalar('MaxVal/Precision', Precision_TrainVal, epoch)
                    writer.add_scalar('MaxVal/Recall', Recall_TrainVal, epoch)
                    writer.add_scalar('MaxVal/F1', F1_TrainVal, epoch)

                    wandb.log({'MaxVal/Accuracy': valAcc_max,
                               'MaxVal/F1': F1_TrainVal,
                               'MaxVal/Precision': Precision_TrainVal,
                               'MaxVal/Recall': Recall_TrainVal,
                               })

                    timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
                    with open(log_txt, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                        f.write(
                            "当前模型最优\t时间：" + str(timestamp) + "\tepoch:" + str(
                                epoch) + "\t" + "MaxVal/Accuracy:" + str(
                                Accuracy_TrainVal * 100) + "\t" + "MaxVal/Precision:" + str(
                                Precision_TrainVal * 100) + "\t" + "MaxVal/Recall:" + str(
                                Recall_TrainVal * 100) + "\t" + "MaxVal/F1:" + str(F1_TrainVal * 100) + "\n" +
                            str(confusion_matrix_TrainVal) + "\n" + "-" * 150 + "\n")
                    # 删除已保存的所有模型，再保存一个最优的模型
                    for file_name in listdir(os.path.join(inRun_dir, 'models')):
                        if file_name.endswith('_maxVal.pth.tar'):
                            os.remove(os.path.join(inRun_dir, 'models', file_name))
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'opt_dict': optimizer.state_dict(),
                    }, os.path.join(inRun_dir, 'models',
                                    config.modelName + '_epoch-' + str(epoch) + '_acc-' + str(valAcc_max)[
                                                                                          :6] + '_maxVal.pth.tar'))
                    print("Max Accuracy Save at {}\n".format(
                        os.path.join(inRun_dir, 'models',
                                     config.modelName + '_epoch-' + str(epoch) + '_acc-' + str(valAcc_max)[
                                                                                           :6] + '_maxVal.pth.tar')))

            print("\n[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, epochAll, epoch_loss, epoch_acc))
            print("[CM] Accuracy_TrainVal:{} Precision_TrainVal:{} Recall_TrainVal:{} F1_TrainVal:{}".format(
                Accuracy_TrainVal, Precision_TrainVal, Recall_TrainVal, F1_TrainVal))
            timeTrainvalStop = time.clock()
            TrainvalTime = timeTrainvalStop - timeTrainvalStart
            TrainvalTime_m, TrainvalTime_s = divmod(TrainvalTime, 60)
            print("当前",
                  str(phase) + "第" + str(epoch) + "epoch，使用" + str(TrainvalTime_m) + "分" + str(TrainvalTime_s) + "秒")

        # 定期保存epoch 已删除。修改为保存为最佳ACC的epoch权重且删除之前的权重
        # if epoch % save_epoch == (save_epoch - 1):
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'opt_dict': optimizer.state_dict(),
        #     }, os.path.join(inRun_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
        #     print("模型保存在{}\n".format(
        #         os.path.join(inRun_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))
        #     pass

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            timeTestStart = time.clock()

            running_loss = 0.00
            running_corrects = 0.00
            LABEL = np.array([])  # 定义一个数组存放混淆矩阵中的真实值
            PRED = np.array([])  # 定义一个数组存放混淆矩阵中的预测值
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
                LABEL, PRED)
            timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
            with open(log_txt, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'写之前会清空文件中的原有数据！
                f.write("测试集推理结果：\n" +
                        str(timestamp) + "\t" + str(epoch) + "\t" + "Accuracy_test:" + str(
                    Accuracy_test * 100) + "\t" + "Precision_test:" + str(
                    Precision_test * 100) + "\t" + "Recall_test:" + str(
                    Recall_test * 100) + "\t" + "F1_test:" + str(F1_test * 100) + "\n" + "混淆矩阵：" + "\n" +
                        str(confusion_matrix_test) + "\n" + str(tabel_test) + "\n")
            if Accuracy_test > testAcc_max:
                testAcc_max = Accuracy_test
                print("MaxTest Accuracy is :", testAcc_max, '\n')
                writer.add_scalar('MaxTest/Accuracy', Accuracy_test, epoch)
                writer.add_scalar('MaxTest/Precision', Precision_test, epoch)
                writer.add_scalar('MaxTest/Recall', Recall_test, epoch)
                writer.add_scalar('MaxTest/F1', F1_test, epoch)

                wandb.log({'MaxTest/Accuracy': testAcc_max,
                           'MaxTest/F1': F1_test,
                           'MaxTest/Precision': Precision_test,
                           'MaxTest/Recall': Recall_test,
                           })

                timestamp = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
                with open(log_txt, 'a', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
                    f.write(
                        "当前模型最优\t时间：" + str(timestamp) + "\tepoch:" + str(epoch) + "\t" + "MAX_Accuracy_test:" + str(
                            Accuracy_test * 100) + "\t" + "MAX_Precision_test:" + str(
                            Precision_test * 100) + "\t" + "MAX_Recall_test:" + str(
                            Recall_test * 100) + "\t" + "MAX_F1_test:" + str(F1_test * 100) + "\n" +
                        str(confusion_matrix_test) + "\n" + "-" * 150 + "\n")
                # 删除已保存的所有模型，再保存一个最优的模型
                for file_name in listdir(os.path.join(inRun_dir, 'models')):
                    if file_name.endswith('_maxTest.pth.tar'):
                        os.remove(os.path.join(inRun_dir, 'models', file_name))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(inRun_dir, 'models',
                                config.modelName + '_epoch-' + str(epoch) + '_acc-' + str(testAcc_max)[
                                                                                      :6] + '_maxTest.pth.tar'))
                print("Max Accuracy Save at {}\n".format(
                    os.path.join(inRun_dir, 'models',
                                 config.modelName + '_epoch-' + str(epoch) + '_acc-' + str(testAcc_max)[
                                                                                       :6] + '_maxTest.pth.tar')))

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('Test/Accuracy', Accuracy_test, epoch)
            writer.add_scalar('Test/Precision', Precision_test, epoch)
            writer.add_scalar('Test/Recall', Recall_test, epoch)
            writer.add_scalar('Test/F1', F1_test, epoch)
            writer.add_scalar('Test/Loss', epoch_loss, epoch)
            writer.add_scalar('Test/LR', optimizer.param_groups[0]['lr'], epoch)

            wandb.log({'Test/Accuracy': Accuracy_test,
                       'Test/F1': F1_test,
                       'Test/Precision': Precision_test,
                       'Test/Recall': Recall_test,
                       'Test/Loss': epoch_loss, 'Test/epoch': epoch,
                       'Test/LR': optimizer.param_groups[0]['lr'],
                       })

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, epochAll, epoch_loss, epoch_acc))
            print("[CM] Accuracy_test:{} Precision_test:{} Recall_test:{} F1_test:{}".format(Accuracy_test,
                                                                                             Precision_test,
                                                                                             Recall_test, F1_test))
            timeTeststop = time.clock()
            timeTest = timeTeststop - timeTestStart
            timeTest_m, timeTest_s = divmod(timeTest, 60)
            print("当前测试" + "第" + str(epoch) + "epoch，使用" + str(timeTest_m) + "分" + str(timeTest_s) + "秒")

        timeEpochEnd = time.clock()
        epochTime = timeEpochEnd - timeEpochStart
        epochTime_m, epochTime_s = divmod(epochTime, 60)
        print("Epoch" + str(epoch) + "已结束，共使用" + str(epochTime_m) + "分" + str(epochTime_s) + "秒")
        scheduler.step()
        print("-" * 150)
    writer.close()


if __name__ == '__main__':
    # https://github.com/fusang1337/Fire-Detection-Base-3DCNN
    # 此代码基于https://github.com/jfzhang95/pytorch-video-recognition重构，用于烟火识别
    # 作者：宋俊猛 邮箱：1@vn.mk

    device = config.device
    print("程序开始执行，当前使用:", str(device), "运算")

    # 定义当前程序运行目录
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # 删除run目录下空文件夹
    del_folders = os.listdir(os.path.join(root_dir, 'run'))
    for folder in del_folders:
        # 将上级路径path与文件夹名称folder拼接出文件夹的路径
        folder2 = os.listdir(os.path.join(root_dir, 'run') + '\\' + folder)
        if folder2 == []:
            # 并将此空文件夹删除
            os.rmdir(os.path.join(root_dir, 'run') + '\\' + folder)
            # print("删除了", str(os.path.join(root_dir,'run') + '\\' + folder))

    # 定义程序保存的文件的目录
    inRun_dir = os.path.join(root_dir, 'run',
                             datetime.now().strftime('%Y%m%d-%H%M%S') + '_' + config.modelName + '_' + str(
                                 config.learnRate) + '_' + str(config.batchSize) + '_' + socket.gethostname())
    if not os.path.exists(inRun_dir):
        os.mkdir(inRun_dir)
        print("当前训练日志文件保存在：", str(inRun_dir))

    wandbconfig = dict(
        learning_rate=config.learnRate,
        batch_size=config.batchSize,
        gamma=config.gamma,
        weightDecay=config.weightDecay,
        modelName=config.modelName,
        epochResume=config.epochResume,
        dataset_dir=config.dataset_dir,
        infra=socket.gethostname(),
    )

    wandb.init(
        project="Fire-Detection-Base-3DCNN",
        name=config.runName + "_" + config.modelName + "_" + str(config.learnRate) + "_" + str(
            config.gamma) + "_" + str(
            config.learnRate) + "_" + str(socket.gethostname()),
        tags=[config.modelName, "LR:" + str(config.learnRate), "BZ:" + str(config.batchSize),
              str(socket.gethostname())],
        config=wandbconfig,
        dir=inRun_dir, group="experiment_1"
    )

    if config.modelName == 'C3D':
        model = C3D_model.C3D(num_classes=config.numClasses, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': config.learnRate},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': config.learnRate * 10}]
    elif config.modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=config.numClasses, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': config.learnRate},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': config.learnRate * 10}]
    elif config.modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=config.numClasses, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif config.modelName == 'VGG':
        model = T2CC3D_model.vgg_3D(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'DIY':
        model = DIY.DIY(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'R2Plus1D':
        model = R2Plus1D_atten.R2Plus1D(num_classes=config.numClasses, layer_sizes=(3, 4, 6, 3))
        train_params = model.parameters()
    elif config.modelName == 'vgg_3D':
        model = T2CC3D_model.vgg_3D(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'vgg11_3d':
        model = vgg11_3d.vgg11_3d(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 't2CC3D':
        model = T2CC3D_model.t2CC3D(num_classes=config.numClasses, pretrained=True)
        train_params = model.parameters()
    elif config.modelName == 'C3D_AttNAtt':
        model = C3D_AttNAtt.C3D_AttNAtt2(sample_size=224, num_classes=config.numClasses,
                                         lstm_hidden_size=512, lstm_num_layers=3)
        train_params = model.parameters()
    elif config.modelName == 'P3D199':
        model = p3d_model.P3D199(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'resnet10':
        model = resnet_3d.resnet10(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'Xception':
        model = xception_3D.Xception(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'mobilenetv2':
        model = mobilenetv2.get_model(num_classes=config.numClasses, sample_size=224, width_mult=1.)
        train_params = model.parameters()
    elif config.modelName == 'ShuffleNetV2':
        model = shufflenetv2.get_model(num_classes=config.numClasses, sample_size=224, width_mult=1.)
        train_params = model.parameters()
    elif config.modelName == 'SqueezeNet':
        model = squeezenet.get_model(num_classes=config.numClasses, sample_size=224, width_mult=1.)
        train_params = model.parameters()
    elif config.modelName == 'mobilenetv3_large':
        model = MobileNetV3.get_model(num_classes=config.numClasses, sample_size=224, width_mult=1.)
        train_params = model.parameters()
    elif config.modelName == 'DIY_X':
        model = DIY_X.Xception(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'GoogLeNet':
        model = GoogLeNet.GoogLeNet(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'VGG3D_Att':
        model = VGG3D_AttNatt.VGG3D_Att(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'F_6M3DC':
        model = F_6M3DC.F_6M3DC(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'Dahan_3D':
        model = Dahan_3D.Dahan_3D(num_classes=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'slowfast_resnet34':
        model = slowfastnet_small.slowfast_resnet34(class_num=config.numClasses)
        train_params = model.parameters()
    elif config.modelName == 'TSM_WITH_ACTION':
        model = TSN_model.TSN(3, 16, 'RGB',
                              is_shift=True,
                              partial_bn=True,
                              base_model="resnet50",
                              shift_div=4,
                              dropout=0.5,
                              img_feature_dim=224)
        train_params = model.parameters()
    elif config.modelName == 'TSM':
        model = TSN_model.TSN(3, 2, 'RGB',
                              base_model="resnet50",
                              dropout=0.5,
                              img_feature_dim=224,
                              partial_bn=True,
                              is_shift=False, shift_div=8)
        train_params = model.parameters()
    elif config.modelName == 'DenseNet_3D':
        model = DenseNet_3D.DenseNet(num_init_features=64,
                                     growth_rate=32,
                                     # block_config=(6, 12, 24, 16))
                                     block_config=(32, 64, 128))
        train_params = model.parameters()

    else:
        print('请检查modelName参数.')
        raise NotImplementedError
    print("模型已创建")
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification

    # 2022年3月30日 已测试不收敛
    # optimizer = optim.SGD(train_params, lr=config.learnRate, momentum=0.9, weight_decay=5e-4)
    # 测试可以收敛 Loss下降慢，抖动严重
    # optimizer = optim.Adam(train_params, lr=config.learnRate, weight_decay=0.0001)
    # optimizer = torch.optim.Adadelta(train_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    optimizer = optim.Adam(train_params, lr=config.learnRate)
    # optimizer = torch.optim.Adadelta(train_params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) optimizer =
    # torch.optim.Adagrad(train_params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    # optimizer = torch.optim.RMSprop(train_params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
    # centered=False) optimizer = torch.optim.SGD(train_params, lr=0.02, momentum=0.95) optimizer = torch.optim.SGD(
    # train_params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4,
    #                                       gamma=0.95)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=config.gamma)  # the scheduler divides the lr by 10 every 10 epochs

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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    criterion.to(device)

    if config.epochResume == 0:
        # print("Training {} from scratch...".format(config.modelName))
        print("新开始训练 {} 模型...".format(config.modelName))
    else:
        checkpoint = torch.load(config.resumePth_path, map_location=lambda storage, loc: storage)
        print("Initializing weights from: {}...".format(config.resumePth_path))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    writer = SummaryWriter(inRun_dir)
    log_txt = inRun_dir + "/" + "train_log.txt"
    with open(log_txt, 'w', encoding='utf-8') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(
            "当前模型超参数：" + config.runName + " Epochs:" + str(config.epochAll) + "\t" + "lr:" + str(
                config.learnRate) + "\t" + "weight_decay :" + str(
                config.weightDecay) + "\t" + "modelName:" + str(config.modelName) + "\t" + "batchSize:" + str(
                config.batchSize) + "\n")

    runModel_dir = os.path.join(inRun_dir, 'models')
    if not os.path.exists(runModel_dir):
        os.mkdir(runModel_dir)

    rootDir_wandb = os.path.join(inRun_dir, 'wandb')
    for dirName in os.listdir(rootDir_wandb):
        print()
        runPy_dir = os.path.join(os.path.join(rootDir_wandb, str(dirName), "files"))
        print("程序文件备份在:", runPy_dir)
        shutil.copy('dataloaders/dataset.py', runPy_dir)
        shutil.copy('train.py', runPy_dir)
        shutil.copy('config.py', runPy_dir)
        shutil.copytree('network', os.path.join(runPy_dir, "network"))

    # 关于num_workers
    # https://blog.csdn.net/JustPeanut/article/details/119146148 windows与linux的num_workers执行有差异
    # num_workers的经验设置值是自己电脑/服务器的CPU核心数
    # num_workers=4*num_gpu
    # https://zhuanlan.zhihu.com/p/479012482

    if config.useLMDB:
        train_dataloader = DataLoader(DatasetFromLMDB(config.lmdb_train, split='train'),
                                      batch_size=config.batchSize, num_workers=config.numWorker)
        val_dataloader = DataLoader(DatasetFromLMDB(config.lmdb_val, split='val'),
                                    batch_size=config.batchSize, num_workers=config.numWorker)
        test_dataloader = DataLoader(DatasetFromLMDB(config.lmdb_test, split='test'),
                                     batch_size=config.batchSize, num_workers=config.numWorker)
    else:
        train_dataloader = DataLoader(VideoDataset(split='train', clip_len=16), batch_size=config.batchSize,
                                      shuffle=True, num_workers=config.numWorker)
        val_dataloader = DataLoader(VideoDataset(split='val', clip_len=16), batch_size=config.batchSize,
                                    num_workers=config.numWorker)
        test_dataloader = DataLoader(VideoDataset(split='test', clip_len=16), batch_size=config.batchSize,
                                     num_workers=config.numWorker)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    print("trainval_sizes:", str(trainval_sizes), "   test_size:", str(test_size))
    main(inRun_dir=inRun_dir, log_txt=log_txt, resume_epoch=config.epochResume, resume_pth=config.resumePth_path,
         epochAll=config.epochAll, test_interval=config.testInterval, useTest=config.useTest)
