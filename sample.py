import os
import numpy as np
import cv2
from PIL import Image

pic_path = "bg_pic"
x = 1
for filename in os.listdir(pic_path):
    print(filename)
    bg = cv2.imread(os.path.join(pic_path, filename))
    if type(bg) is np.ndarray and len(bg.shape) == 3 and bg.shape[0] > 100 and bg.shape[1] > 100:
        bg = bg
    else:
        continue
    bg_resize = cv2.resize(bg, (300, 300))
    bg_resize = cv2.cvtColor(bg_resize, cv2.COLOR_BGR2RGB)
    name = np.random.randint(1, 21)
    img_font = cv2.imread("yellow/{}.png".format(name), cv2.IMREAD_UNCHANGED)
    ran_w = np.random.randint(50, 180)
    img_new = cv2.resize(img_font, (ran_w, ran_w))
    img_new = cv2.cvtColor(img_new, cv2.COLOR_BGRA2RGBA)
    ran_x1 = np.random.randint(0, 300 - ran_w)
    ran_y1 = np.random.randint(0, 300 - ran_w)

    bg_resize = Image.fromarray(bg_resize)
    img_new = Image.fromarray(img_new)
    r, g, b, a = img_new.split()
    bg_resize.paste(img_new, (ran_x1, ran_y1), mask=a)

    ran_x2 = ran_x1 + ran_w
    ran_y2 = ran_y1 + ran_w
    for i in range(1, 9001):
        bg_resize.save("data/{0}{1}.png".format(x, "." + str(ran_x1) + "." + str(ran_y1) +
                                                "." + str(ran_x2) + "." + str(ran_y2)))
