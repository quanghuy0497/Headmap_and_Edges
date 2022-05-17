import numpy as np
import cv2 as cv
import sys
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from einops import rearrange

image_path = str(sys.argv[1])
img_color = cv.imread(image_path, 1)
img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)
img = cv.imread(image_path, 0)
edges = cv.Canny(img,80,200)

tensor = transforms.ToTensor()
img = tensor(img).squeeze()
edges = tensor(edges).squeeze()

edges = img + 2 * edges

print(edges.shape)
plt.subplot(121), plt.imshow(img_color)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
