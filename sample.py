from PIL import Image
import numpy as np
# import np.set_printoptions
import torch

im_in = np.array(Image.open('movies/coco_AutoEncoder_in_sample_0702.png'))
im_out = np.array(Image.open('movies/coco_AutoEncoder_out_sample_0702.png'))

images = torch.Tensor(im_in)
outputs = torch.Tensor(im_out)

# images = torch.Tensor(Image.open('movies/coco_AutoEncoder_in_sample_0702.png'))
# outputs = torch.Tensor(Image.open('movies/coco_AutoEncoder_out_sample_0702.png'))

# tmp = im_in**2 - im_out**2
# tmp = im_in_t**2 - im_out_t**2

criterion = torch.nn.MSELoss()
loss = criterion(outputs, images)

# tmp2 = np.average(tmp)
# tmp2 = torch.mean(tmp)

# print(tmp2)
print(loss)

np.set_printoptions(threshold=np.inf)
print(im_out)