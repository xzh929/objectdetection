from net2 import ResNet18
from dataset import Yellowdataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from iou import IOU
import torch
import numpy as np
import os
from torch.nn.functional import one_hot

train_dataset = Yellowdataset("train_data2")
train_loader = DataLoader(train_dataset, batch_size=45, shuffle=True)
test_dataset = Yellowdataset("test_data")
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=True)

net = ResNet18().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists("module/yellow2.pth"):
    net.load_state_dict(torch.load("module/yellow2.pth"))
    print("load module!")
else:
    print("no module!")


def main():
    global out11, tag11
    iou = 0.
    for epoch in range(1000):
        sum_train_loss = 0.
        sum_iou = 0.
        x = 1
        sum_score = 0.
        for i, (img, tag1, tag2) in enumerate(train_loader):
            net.train()
            img, tag1, tag2 = img.cuda(), tag1.cuda(), tag2.cuda()
            tag2 = one_hot(tag2, 2)
            tag2 = tag2.squeeze()
            out1, out2 = net(img)  # out1为坐标，out2为正负

            loss1 = loss_func(out1, tag1)
            tag2 = tag2.to(torch.float32)
            loss2 = loss_func(out2, tag2)
            loss = loss1 + loss2
            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_train_loss += loss.item()
            out1 = out1.detach().cpu()
            tag1 = tag1.detach().cpu()
            del_index = []
            for y in range(45):
                if tag1[y].equal(torch.zeros(4)):
                    del_index.append(y)
            out11 = np.delete(out1, del_index, axis=0)
            tag11 = np.delete(tag1, del_index, axis=0)
            out11, tag11 = out11.numpy(), tag11.numpy()
            sum_iou += np.mean(IOU(out11, tag11))
            pre = torch.argmax(out2, dim=1)
            label = torch.argmax(tag2, dim=1)
            score = torch.mean(pre.eq(label).float()).item()
            sum_score += score

        avg_train_loss = sum_train_loss / len(train_loader)
        avg_train_iou = sum_iou / len(train_loader)
        avg_score = sum_score / len(train_loader)
        print("epoch:{} train_loss:{} train_iou:{} score:{}".format(epoch, avg_train_loss, avg_train_iou, avg_score))
        if avg_train_iou > iou:
            iou = avg_train_iou
            torch.save(net.state_dict(), "module/yellow2.pth")
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
