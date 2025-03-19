import torch
import numpy as np
from network import C3D_model, R2Plus1D_model, R3D_model
import cv2

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open(r'dataloaders\fire_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model

    model = R3D_model.R3DClassifier(num_classes=3, layer_sizes=(2, 2, 2, 2))
    # model = C3D_model.C3D(num_classes=3)
    checkpoint = torch.load(
        r'F:\THHI\program\pytorch-video-recognition-ubuntu\run\run_0\models\R3D-ucf101_epoch-38.pth.tar',
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    # video = r'F:\ubuntu\program\data\ccdandnir\none/01_1_ALARM_CCD-NIR_20220307-152839_1.avi'
    # video = r'F:\ubuntu\program\data\ccdandnir\smoke/03_2_ALARM_CCD-NIR_20220308-144318_5.avi'
    # video = r"F:\ubuntu\program\data\近红外和可见光\第5次配准结果/ALARM_CCD-NIR_20220307-152623.avi"
    video = r"http://192.168.20.11:8080/video"
    video = r"D:\网上收集烟雾/Dry_leaf_smoke_02.avi"

    cap = cv2.VideoCapture(video)
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)

        cv2.imshow('result', frame)
        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
