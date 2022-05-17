import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
import cv2 as cv

class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size = 3, stride=2, padding=1), 
                    nn.BatchNorm2d(64), nn.ReLU(), 
                    nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1), 
                    nn.BatchNorm2d(128), nn.ReLU(), 
                    nn.Conv2d(128, 256, kernel_size = 3, stride=2, padding=1), 
                    nn.BatchNorm2d(256), nn.ReLU())
        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size = 3, stride=2, padding=1),
                    nn.BatchNorm2d(128), nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size = 3, stride=2, padding=1),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.ConvTranspose2d(64, 3, kernel_size = 3, stride=2, padding=1),
                    nn.BatchNorm2d(3), nn.ReLU())
    
    def forward(self, x):
        '''
        x     =   [b, 3, h, w] or [b, n, 3, h, w]
        out   =   same as x
        '''
        if len(x.shape) == 5:
            d1, d2 = x.shape[0], x.shape[1]
            x = rearrange(x, 'b n c h w -> (b n) c h w')
            out = self.deconv(self.conv(x))
            return rearrange(out, '(b n) c h w -> b n c h w', b = d1, n = d2)
        else:
            return self.deconv(self.conv(x))

model = FCN()
tensor = transforms.ToTensor()
# x = torch.rand(21, 5, 3, 84, 84)
# print(model(x).shape)

image_path = str(sys.argv[1])
pil_img = PIL.Image.open(image_path)
cv_image = np.array(pil_img)

# Heatmap
img_tensor = tensor(pil_img).unsqueeze(0)
heatmap = model(img_tensor)

# Canny Edge
edges = cv.Canny(cv_image, 80, 200)

# Process image for showing
cv_image = rearrange(tensor(cv_image).squeeze(), 'c h w -> h w c')
grey_weight = torch.tensor([0.2989, 0.5870, 0.1140])
gray_cv_image = tensor(np.dot(cv_image, grey_weight))       # convert color to greyscale

edges = tensor(edges).squeeze()
edges = gray_cv_image + 2 * edges

img_tensor = rearrange(img_tensor.squeeze(), 'c h w -> h w c')
heatmap = rearrange(heatmap.squeeze(), 'c h w -> h w c')
edges = rearrange(edges, 'c h w -> h w c')

with torch.no_grad():
    plt.subplot(131), plt.imshow(img_tensor)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(heatmap)
    plt.title('Heatmap'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(edges, cmap='gray')
    plt.title('Edges map'), plt.xticks([]), plt.yticks([])
    plt.show()
