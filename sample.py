from PIL import Image
import numpy as np

im_in = np.array(Image.open('movies/coco_AutoEncoder_in_sample_0702.png'))
im_out = np.array(Image.open('movies/coco_AutoEncoder_out_sample_0702.png'))

tmp = im_in**2 - im_out**2

tmp2 = np.average(tmp)

print(tmp2)