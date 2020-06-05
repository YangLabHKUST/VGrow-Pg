import tensorflow as tf
import numpy as np

################################ visualizing image grids ################################

def montage(images, grid):

    s = np.shape(images)
    assert s[0] == np.prod(grid) and np.shape(s)[0] == 4
    bigimg = np.zeros((s[1]*grid[0], s[1]*grid[1], s[3]), dtype=np.float32)

    for i in range(grid[0]):
        for j in range(grid[1]):
            bigimg[s[1] * i : s[1] * i + s[1], s[1] * j : s[1] * j + s[1]] += images[grid[1] * i + j]

    return np.rint(bigimg*255).clip(0, 255).astype(np.uint8)

################################ pre-processing real images ################################

def downscale(img):
    s = img.shape
    out = np.reshape(img, [-1, s[1]//2, 2, s[2]//2, 2, s[3]])
    return np.mean(out, axis=(2, 4))

def upscale(img):
    return np.repeat(np.repeat(img, 2, axis=1), 2, axis=2)

def process_real(x, lod_in):
    y = x / 127.5 - 1
    alpha = lod_in - np.floor(lod_in)
    y = (1 - alpha)*y + alpha*upscale(downscale(y))
    for i in range(int(np.floor(lod_in))):
        y = upscale(y)
    return y

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

