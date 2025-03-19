import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pickle
import torch.utils.data as data
from torch.utils.data import DataLoader

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, dataset='', split='val', labels_path='', clip_len=16):
        folder = os.path.join(dataset, split)
        self.clip_len = clip_len
        self.split = split

        # resize_height和resize_width填写现在的数据集的图片尺寸
        self.resize_height = 224
        self.resize_width = 224

        # 将图片裁剪到crop_size*crop_size的尺寸
        self.crop_size = 224

        # Obtain all the filetrivaltsts of files inside all the class folders 获取所有类文件夹中文件的所有文件名
        # Going through each class folder one at a time 一次浏览一个类文件夹
        self.ftrivaltsts, labels = [], []
        for label in sorted(os.listdir(folder)):
            for ftrivaltst in os.listdir(os.path.join(folder, label)):
                self.ftrivaltsts.append(os.path.join(folder, label, ftrivaltst))
                labels.append(label)

        assert len(labels) == len(self.ftrivaltsts)
        print('Number of {} videos: {:d}'.format(split, len(self.ftrivaltsts)))

        # Prepare a mapping between the label trivaltsts (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        print(self.label2index)
        # Convert the list of label trivaltsts into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists(labels_path):
            with open(labels_path, 'w') as f:
                # with open('dataloaders/ucf_labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.ftrivaltsts)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.ftrivaltsts[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        # 此处涉及均值的计算，原代码直接填充[90.0, 98.0, 102.0, 90]，比较暴力，一般应自己计算得到
        # 计算均值的代码参考 https://github.com/TingsongYu/PyTorch_Tutorial/blob/master/Code/1_data_prepare/1_5_compute_mean.py
        for i, frame in enumerate(buffer):
            # 修改为4通道
            frame = frame - np.array([[[10.44, 82.52, 79.10, 78.16]]])
            frame = frame / np.array([[[50.53, 63.06, 62.59, 62.85]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        imgs = os.listdir(file_dir)
        imgs.sort(key=lambda x: int(x.split(".")[0]))
        frame_count = len(imgs)
        # 适配多通道
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 4), np.dtype('float32'))

        for i, frame_name in enumerate(imgs):
            frame_name = os.path.join(file_dir, frame_name)
            frame = np.array(cv2.imread(frame_name, -1)).astype(np.float64)
            buffer[i] = frame
        return buffer




    # def load_frames(self, file_dir):
    #     frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    #     frames.sort(key=lambda x: int(x.split(".")[0]))
    #     frame_count = len(frames)
    #     # 修改为4通道
    #     buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
    #     # buffer = np.empty((frame_count, self.resize_height, self.resize_width, 4), np.dtype('float32'))
    #     for i, frame_trivaltst in enumerate(frames):
    #         # 读取4通道
    #         frame = np.array(cv2.imread(frame_trivaltst)).astype(np.float64)
    #         # frame = np.array(cv2.imread(frame_trivaltst, -1)).astype(np.float64)
    #         buffer[i] = frame
    #
    #     return buffer

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
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)


class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__trivaltst__ + ' (' + self.db_path + ')'


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


def folder2lmdb(dataset_dir, trivaltst="train", labels_path='', det_lmdb_dir='', map_size=1073741824,
                write_frequency=100):
    directory = osp.expanduser(osp.join(dataset_dir, trivaltst))
    print("Loading dataset from %s" % directory)

    data_loader = DataLoader(VideoDataset(dataset=dataset_dir, split=trivaltst, labels_path=labels_path, clip_len=16),
                             num_workers=0, collate_fn=lambda x: x)

    lmdb_path = osp.join(det_lmdb_dir, "%s.lmdb" % trivaltst)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)

    # map_size设置过大会乱码报错，过小会mapsize limit reached报错
    # Windows会直接把空间占掉，linux不会出现这个问题
    # 1099511627776 = 1T
    # 8589934592 = 8G
    # 1073741824 = 1G
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    # db = lmdb.open(lmdb_path, subdir=isdir)

    txn = db.begin(write=True)
    for idx, data in enumerate(data_loader):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encode('ascii'), dumps_data((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_data(keys))
        txn.put(b'__len__', dumps_data(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


if __name__ == '__main__':
    dataset_dir = r"G:\program\date\2paper\4tongdaobeijingcaifen/"
    trivaltst = "test"
    labels_path = r'G:\program\Fire-Detection-Base-3DCNN\dataloaders\fire_labels.txt'
    det_lmdb_dir = r"G:\program\Fire-Detection-Base-3DCNN\data"
    map_size = 1073741824 * 1
    folder2lmdb(dataset_dir, trivaltst, labels_path, det_lmdb_dir, map_size, 50)
