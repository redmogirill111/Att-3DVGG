import os
import os.path as osp
import lmdb
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import config
import pickle


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
    """

    def __init__(self, split='train', clip_len=16):
        folder = os.path.join(config.dataset_dir, split)
        self.clip_len = clip_len
        self.split = split

        # resize_height和resize_width填写现在的数据集的图片尺寸
        self.resize_height = 224
        self.resize_width = 224

        # 将图片裁剪到crop_size*crop_size的尺寸
        self.crop_size = 224

        # Obtain all the filenames of files inside all the class folders 获取所有类文件夹中文件的所有文件名
        # Going through each class folder one at a time 一次浏览一个类文件夹
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('dataset:{}视频有: {:d}个'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints) 准备标签名称（字符串）和索引（整数）之间的映射
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        print(self.label2index)
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        # print(os.path.join(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.exists('dataloaders/fire_labels.txt'):
            print("dataset:没有发现fire_labels,自动生成")
            with open(r'dataloaders/fire_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def randomflip(self, buffer):
        """以 0.5 的概率随机水平翻转给定图像和GT。"""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer

    def normalize(self, buffer):
        # Todo:此处涉及均值的计算，原代码直接填充[90.0, 98.0, 102.0, 90]，比较暴力，一般应自己计算得到,建议优化
        # 计算均值的代码参考 https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/1_data_prepare/1_5_compute_mean.py
        # normMean = [0.41534585, 0.41007122, 0.38520297]
        # normStd = [0.24997446, 0.2495269, 0.2608023]
        # 2022年9月2日 帧差+RGB数据集
        # normMean = [0.041900594, 0.32369658, 0.30989408, 0.3061049]
        # normStd = [0.20036174, 0.24826877, 0.24610789, 0.24721156]
        # normMean = [10.443644399424354, 82.52525698884307, 79.1075393418196, 78.16661254113843]
        # normStd = [50.5377048698452, 63.06563264343568, 62.592513704508285, 62.856391636821456]

        for i, frame in enumerate(buffer):
            # 适配多通道
            if config.channel == 4:
                frame = frame - np.array(
                    [[[10.443644399424354, 82.52525698884307, 79.1075393418196, 78.16661254113843]]])
                frame = frame / np.array(
                    [[[50.5377048698452, 63.06563264343568, 62.592513704508285, 62.856391636821456]]])
            else:
                frame = frame - np.array([[[105.523094, 104.26006, 98.04103]]])
                frame = frame / np.array([[[63.678574, 63.634727, 66.397]]])
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    # 2022年8月31日 修复了Windows下文件顺序读取错误的问题
    def load_frames(self, file_dir):
        imgs = os.listdir(file_dir)
        imgs.sort(key=lambda x: int(x.split(".")[0]))
        frame_count = len(imgs)
        # 适配多通道
        if config.channel == 4:
            buffer = np.empty((frame_count, self.resize_height, self.resize_width, 4), np.dtype('float32'))
        else:
            buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(imgs):
            frame_name = os.path.join(file_dir, frame_name)
            if config.channel == 4:
                frame = np.array(cv2.imread(frame_name, -1)).astype(np.float64)
            else:
                frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        return buffer

    # 以下是原load_frames，读取的jpg文件顺序不对（1，10，11，12---19，2，20，21）
    # def load_frames(self, file_dir):
    # frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    # frame_count = len(frames)
    # # 适配多通道
    # if config.channel == 4:
    #     buffer = np.empty((frame_count, self.resize_height, self.resize_width, 4), np.dtype('float32'))
    # else:
    #     buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
    # for i, frame_name in enumerate(frames):
    #     # 适配多通道
    #     if config.channel == 4:
    #         frame = np.array(cv2.imread(frame_name, -1)).astype(np.float64)
    #     else:
    #         frame = np.array(cv2.imread(frame_name)).astype(np.float64)
    #     buffer[i] = frame
    # return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if (buffer.shape[0] - clip_len) <= 0:
            print(buffer.shape[0] - clip_len)
            print("dataset:模型要求一个样本至少", str(clip_len), "张图，当前样本有", str(buffer.shape[0]), "张图")

        # 此处low >= high可能是因为网络要求至少有16张图片，而 buffer=(15, 128, 171, 3)，因为某些文件夹里面只有16张以下图片。
        #
        # 5行代码找出不安分的小鬼
        # import os
        # number = 0
        # for root, dirs, files in os.walk(r"数据集目录"):
        #     if len(files)  <= 17 and "ALARM"in root:
        #         print(root)

        # Todo:此处的裁剪方法有问题，随机裁剪可能会导致将目标裁掉，建议寻找其他方案
        # 随机选择起始索引以裁剪视频
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # 使用索引裁剪和抖动视频。 空间裁剪在整个数组，所以每一帧都在同一位置裁剪。 时间的抖动通过选择连续帧来发生
        if (buffer.shape[1] != crop_size) or (buffer.shape[2] != crop_size):
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            buffer = buffer[time_index:time_index + clip_len,
                     height_index:height_index + crop_size,
                     width_index:width_index + crop_size, :]
        else:
            buffer = buffer[time_index:time_index + clip_len,
                     :,
                     :, :]
        return buffer


def loads_data(buf):
    return pickle.loads(buf)


# 读取LMDB数据库程序来自Efficient-PyTorch https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
# 原程序不能多线程，出现TypeError: can't pickle Environment objects
# 解决方案来自 https://github.com/pytorch/vision/issues/689
# Todo:__init__不能有lmdb.open出现，否则多线程报错，因此没办法得到self.length参数，暂时手动得到self.length的值，在__len__(self)中返回.
class DatasetFromLMDB(Dataset):
    def __init__(self, db_path, split='train', transform=None, target_transform=None):
        """do not open lmdb here!!"""
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        # # 注释以下行，修复 can't pickle Environment objects
        # self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
        #                      readonly=True, lock=False,
        #                      readahead=False, meminit=False)
        # with self.env.begin(write=False) as txn:
        #     self.length = loads_data(txn.get(b'__len__'))
        #     self.keys = loads_data(txn.get(b'__keys__'))
        # self.env.close()
        # print("dataset: len:", self.length)

    def open_lmdb(self):
        self.env = lmdb.open(self.db_path, subdir=osp.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

    def __getitem__(self, item: int):
        if not hasattr(self, 'txn'):
            self.open_lmdb()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[item])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        # buf = six.BytesIO()
        # buf.write(imgbuf)
        # buf.seek(0)
        # img = Image.open(buf).convert('RGB')
        img = imgbuf

        # load label
        target = unpacked[1]

        if self.transform is not None:
            # img = self.transform(img)
            pass

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        # return self.length
        if self.split == "train":
            return 7607
        elif self.split == "val":
            return 3255
        else:
            return 2360

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_data(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pickle.dumps(obj)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_data = VideoDataset(split='train', clip_len=8)
    train_loader = DataLoader(train_data, batch_size=20, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
