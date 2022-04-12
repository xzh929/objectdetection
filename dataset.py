from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms as T
import torch
import numpy as np
from PIL import Image


class Yellowdataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.filename = os.listdir(data_path)
        self.transforms = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, item):
        label = torch.Tensor(np.array(self.filename[item].split(".")[1:5], dtype=np.float32) / 300)
        img = Image.open(os.path.join(self.path, self.filename[item]))
        img = self.transforms(img)
        return img, label


if __name__ == '__main__':
    data = Yellowdataset("data")
    a = data[0]
    print(a[0].shape, a[1].shape)
    img = T.ToPILImage()(a[0])
    img.show()
