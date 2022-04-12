from dataset import Yellowdataset
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from iou import IOU
import torch
import numpy as np
from ResNet import ResNet18
from torchvision import transforms as T
from PIL import ImageDraw, Image
import os

test_dataset = Yellowdataset("test_data")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

net = ResNet18().cuda()
opt = optim.Adam(net.parameters())
loss_func = nn.MSELoss()

if os.path.exists("module/yellow.pth"):
    net.load_state_dict(torch.load("module/yellow.pth"))
    print("load module!")

def main():
    for i, (img, tag) in enumerate(test_loader):
        net.train()
        img, tag = img.cuda(), tag.cuda()
        out = net(img)

        loss = loss_func(out, tag)
        out = out.detach().cpu().numpy() * 300
        tag = tag.detach().cpu().numpy() * 300
        iou = np.mean(IOU(out, tag))
        print("epoch:{} test_loss:{} test_iou{}".format(i, loss.item(), iou))
        img_pic = T.ToPILImage()(img[0])
        draw = ImageDraw.Draw(img_pic)
        draw.rectangle(np.array(tag[0]), outline="red", width=2)
        draw.rectangle(np.array(out[0]), outline="yellow", width=2)
        img_pic.show()


if __name__ == '__main__':
    main()