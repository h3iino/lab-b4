from PIL import Image
import numpy as np
import torch

im_in = np.array(Image.open('movies/coco_AutoEncoder_in_sample_0702.png'))
im_out = np.array(Image.open('movies/coco_AutoEncoder_out_sample_0702.png'))

im_in_t = torch.Tensor(im_in)
im_out_t = torch.Tensor(im_out)

# tmp = im_in**2 - im_out**2
tmp = im_in_t**2 - im_out_t**2

tmp2 = np.average(tmp)
tmp2 = torch.average(tmp)

print(tmp2)