import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
import torch
import sys
from canny_edge_detector import CannyEdgeDetector
import numpy as np
from einops import rearrange
import pdb
from torchvision import transforms
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Get the image.
image_path = str(sys.argv[1])
img = PIL.Image.open(image_path)
tensor = transforms.PILToTensor()
# img = np.asarray(img)
# print(img)

# gray_img = tensor(img)[...,:3]
# print(gray_img)
img = tensor(img)

img = rearrange(img, 'c h w -> h w c')[..., :3]


grey_weight = torch.tensor([0.2989, 0.5870, 0.1140])
gray_img = np.dot(img, grey_weight)
plt.imshow(img, cmap='gray')
plt.show()

edge_detector = CannyEdgeDetector(gray_img, dimension=1, sigma=2)
final_image = edge_detector.detect_edges()
plt.imshow(final_image, cmap='gray')
plt.show()


# def CannyConvert(x, dimension, sigma):
#     b, n, _, _ = x.shape
#     x_input = rearrange(x, 'b n h w -> (b n) h w')
#     for i in range(x_input.shape[0]):
#         edge_detector = CannyEdgeDetector(x_input[i], dimension=dimension, sigma=sigma)
#         x_input[i] = edge_detector.detect_edges()
#     x_output = rearrange(x_input, '(b n) h w -> b n h w', b = b, n = n)
#     return x_output

# x = torch.rand(21, 5, 3, 84, 84)
# x.to(device)
# x = rearrange(x, 'b n c h w -> b n h w c')
# grey_weight = torch.tensor([0.2989, 0.5870, 0.1140])
# gray_x = np.dot(x, grey_weight)
# # gray_x = torch.einsum('b n h w c, c -> b n h w', x, grey_weight)
# canny_x = CannyConvert(gray_x, dimension=1, sigma=2)
# print(canny_x.shape)