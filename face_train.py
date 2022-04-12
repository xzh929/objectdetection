from ResNet import ResNet18
from facedataset import FaceDataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from iou import IOU
import numpy as np
import torch

train_dataset = FaceDataset("D:\MTCNN\celeba\Anno\list_bbox_celeba.txt", is_Train=True)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataset = FaceDataset("D:\MTCNN\celeba\Anno\list_bbox_celeba.txt", is_Train=False)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)

net = ResNet18().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()


def main():
    ini_iou = 0.
    for epoch in range(1000):
        sum_train_loss = 0.
        sum_iou = 0.
        for i, (img, tag) in enumerate(train_loader):
            img, tag = img.cuda(), tag.cuda()
            out = net(img)
            loss = loss_func(out, tag)
            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_train_loss += loss.item()
            out = out.detach().cpu().numpy()
            tag = tag.detach().cpu().numpy()
            sum_iou += np.mean(IOU(out, tag))
        avg_train_loss = sum_train_loss / len(train_loader)
        avg_train_iou = sum_iou / len(train_loader)
        print("epoch:{} train_loss:{} train_iou:{}".format(epoch, avg_train_loss, avg_train_iou))
        if avg_train_iou > ini_iou:
            ini_iou = avg_train_iou
            torch.save(net.state_dict(), "module/face.pth")
            print("save success!")


if __name__ == '__main__':
    main()
