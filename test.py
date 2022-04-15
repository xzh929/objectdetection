from dataset import Yellowdataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from iou import IOU
import torch
import numpy as np
# from ResNet import ResNet18
from net2 import ResNet18
from torchvision import transforms as T
from PIL import ImageDraw, Image
import os

test_dataset = Yellowdataset("test_data2")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

net = ResNet18().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists("module/yellow2.pth"):
    net.load_state_dict(torch.load("module/yellow2.pth"))
    print("load module!")


def main():
    for i, (img, tag1, tag2) in enumerate(test_loader):
        net.eval()
        img, tag1, tag2 = img.cuda(), tag1.cuda(), tag2.cuda()
        out1, out2 = net(img)

        # loss = loss_func(out, tag)
        out1 = out1.detach().cpu().numpy() * 300
        tag1 = tag1.detach().cpu().numpy() * 300
        out2 = torch.argmax(out2,dim=1)
        iou = np.mean(IOU(out1, tag1))
        print("epoch:{} test_iou:{} tag:{}".format(i, iou, out2.item()))
        img_pic = T.ToPILImage()(img[0])
        draw = ImageDraw.Draw(img_pic)
        draw.rectangle(np.array(tag1[0]), outline="red", width=2)
        draw.rectangle(np.array(out1[0]), outline="yellow", width=2)
        img_pic.show()
        break


if __name__ == '__main__':
    main()
