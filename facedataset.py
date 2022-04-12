import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch


class FaceDataset(Dataset):
    def __init__(self, path, is_Train=True):
        self.path = path
        self.filename = []
        self.tag = []
        self.is_train = is_Train
        self.transforms = T.Compose([T.ToTensor(), T.Resize((300, 300))])
        i = 0
        with open(self.path, encoding="utf-8") as f:
            f = f.readlines()
            for line in f:
                line = line.split()
                self.filename.append(line[0])
                x, y, w, h = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                self.tag.append([x, y, x + w, y + h])
                i += 1
                if i == 21000:
                    break

        # print(self.filename, self.tag)

    def __len__(self):
        if self.is_train:
            return len(self.filename[0:18000])
        else:
            return len(self.filename[18000:])

    def __getitem__(self, item):
        if self.is_train:
            self.train_filename = self.filename[:18000]
            self.train_tag = self.tag[:18000]
            img = Image.open(os.path.join("D:\MTCNN\celeba\img_celeba.7z\img_celeba", self.train_filename[item]))
            img_resize = self.transforms(img)
            w_factor = 300 / img.size[0]
            h_factor = 300 / img.size[1]
            factor = torch.tensor([w_factor, h_factor, w_factor, h_factor])
            tag = torch.Tensor(np.asarray(self.train_tag[item], dtype=np.float32)) * factor/300
            return img_resize, tag
        else:
            self.test_filename = self.filename[18000:]
            self.test_tag = self.tag[18000:]
            img = Image.open(os.path.join("D:\MTCNN\celeba\img_celeba.7z\img_celeba", self.test_filename[item]))
            img_resize = self.transforms(img)
            w_factor = 300 / img.size[0]
            h_factor = 300 / img.size[1]
            factor = torch.tensor([w_factor, h_factor, w_factor, h_factor])
            tag = torch.Tensor(np.asarray(self.test_tag[item], dtype=np.float32)) * factor / 300
            return img_resize, tag


if __name__ == '__main__':
    dataset = FaceDataset("D:\MTCNN\celeba\Anno\list_bbox_celeba.txt", is_Train=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, (img, tag) in enumerate(loader):
        print(i)
        print(img.shape, tag)
    # a = dataset[0]
    # print(len(dataset))
    # print(a[0].shape, a[1])
