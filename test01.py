from torchvision import transforms as T
from PIL import Image
import torch
from iou import IOU
from torch.nn.functional import one_hot
import numpy as np

# img = Image.open(r"D:\MTCNN\celeba\img_celeba.7z\img_celeba\000001.jpg")
# img = T.ToTensor()(img)
# img = T.Resize((300,300))(img)
# img = T.ToPILImage()(img)
# img.show()
a = torch.randn(3, 2)
c = torch.randn(2)
b = torch.zeros(2)
e = torch.row_stack((a, b))
print(e)
for i in range(4):
    if e[i].equal(b):
        e = np.delete(e, i, axis=0)
print(e)


# print(a, b)
# sum = 1
# for i in range(3):
#     # print(a[:],b[:])
#     if not a[i].equal(b):
#         print(a[i],b)
#         sum += 1
# print(sum)
