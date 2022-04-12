from ResNet import ResNet18
from dataset import Yellowdataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from iou import IOU
import torch
import numpy as np
import os

train_dataset = Yellowdataset("data")
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
test_dataset = Yellowdataset("test_data")
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)

net = ResNet18().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists("module/yellow.pth"):
    net.load_state_dict(torch.load("module/yellow.pth"))
    print("load module!")

def main():
    iou = 0.
    for epoch in range(1000):
        sum_train_loss = 0.
        sum_iou = 0.
        for i, (img, tag) in enumerate(train_loader):
            net.train()
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
        print("epoch:{} train_loss:{} train_iou{}".format(epoch, avg_train_loss, avg_train_iou))
        if avg_train_iou > iou:
            iou = avg_train_iou
            torch.save(net.state_dict(), "module/yellow.pth")
            print("save success!")

        # sum_test_loss = 0.
        # for i, (img, tag) in enumerate(test_loader):
        #     net.eval()
        #     img, tag = img.cuda(), tag.cuda()
        #     out = net(img)
        #     loss = loss_func(out, tag)
        #     print(loss.item())


if __name__ == '__main__':
    main()
