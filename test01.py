from torchvision import transforms as T
from PIL import Image

img = Image.open(r"D:\MTCNN\celeba\img_celeba.7z\img_celeba\000001.jpg")
img = T.ToTensor()(img)
img = T.Resize((300,300))(img)
img = T.ToPILImage()(img)
img.show()
