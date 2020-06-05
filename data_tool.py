import os
import numpy as np
import tensorflow as tf
import glob
import imageio


def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)

def create_mnist_tfr(data_dir, tfrecord_dir='mnist'):
    # "data_dir" should be a folder containing "train-images-idx3-ubyte.gz"

    resolution_log2 = int(np.log2(32))

    # reading raw images
    import gzip
    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28, 1)

    # 28x28 -> 32x32
    images = np.pad(images, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    # saving images for measurement
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    for i in range(10000):
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % i), images[i])

    # creating tfrecords
    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/mnist%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = np.shape(images)[0]
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = images[order[idx]]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def create_fashionmnist_tfr(data_dir, tfrecord_dir='fashionmnist'):
    # "data_dir" should be a folder containing "train-images-idx3-ubyte.gz"

    resolution_log2 = int(np.log2(32))

    # reading raw images
    import gzip
    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28, 1)

    # 28x28 -> 32x32
    images = np.pad(images, [(0,0), (2,2), (2,2), (0,0)], 'constant', constant_values=0)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    # saving images for measurement
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    for i in range(10000):
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % i), images[i])

    # creating tfrecords
    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/fashionmnist%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = np.shape(images)[0]
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = images[order[idx]]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def create_cifar10_tfr(data_dir, tfrecord_dir='cifar10'):
    # "data_dir" should be a folder containing 5 files named from "data_batch_1" to "data_batch_5"

    resolution_log2 = int(np.log2(32))

    # reading raw images
    import pickle
    images = []
    for batch in range(1, 6):
        with open(os.path.join(data_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data['data'].reshape(-1, 3, 32, 32))
    images = np.concatenate(images)
    images = images.transpose(0, 2, 3, 1)

    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    # saving images for measurement
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    for i in range(10000):
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % i), images[i])

    # creating tfrecords
    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/cifar10%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = np.shape(images)[0]
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = images[order[idx]]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def create_celeba_tfr(data_dir, tfrecord_dir='celeba'):
    # "data_dir" should be a folder containing 202599 218*178 png files

    resolution_log2 = int(np.log2(128))

    # reading raw images
    import PIL.Image
    glob_pattern = os.path.join(data_dir, '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    images_num = len(image_filenames)

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    # saving images for measurement
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    for idx in range(10000):
        img = np.asarray(PIL.Image.open(image_filenames[idx]))
        img = img[57:185, 25:153]
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), img)

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/celeba%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
        img = img[57:185, 25:153]
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def create_celeba64_tfr(data_dir, tfrecord_dir='celeba64'):
    # "data_dir" should be a folder containing 202599 218*178 png files

    resolution_log2 = int(np.log2(64))

    # reading raw images
    import PIL.Image
    glob_pattern = os.path.join(data_dir, '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    images_num = len(image_filenames)

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))

    # saving images for measurement
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    for idx in range(10000):
        img = np.asarray(PIL.Image.open(image_filenames[idx]), dtype='int64')
        img = img[57:185, 25:153]
        img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
        quant = np.rint(img).clip(0, 255).astype(np.uint8)
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), quant)

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/celeba64%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]), dtype='int64')
        img = img[57:185, 25:153]
        img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def create_lsun_tfr(data_dir, tfrecord_dir='lsun'):
    # "data_dir" should be a folder containing mdb files

    resolution_log2 = int(np.log2(256))

    # reading raw images
    import lmdb
    import cv2
    import io
    import sys
    import PIL.Image

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals_all'))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/lsun%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    with lmdb.open(data_dir, readonly=True).begin(write=False) as txn:

        count = 0

        for idx, (key, value) in enumerate(txn.cursor()):
            try:
                try:
                    img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                    if img is None:
                        raise IOError('cv2.imdecode failed')
                    img = img[:, :, ::-1] # BGR => RGB
                except IOError:
                    img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                crop = np.min(img.shape[:2])
                img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                img = PIL.Image.fromarray(img, 'RGB')
                img = img.resize((256, 256), PIL.Image.ANTIALIAS)
                img = np.asarray(img)
                imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals_all', '%06d.png' % count), img)
                count += 1
                for lod, tfr_writer in enumerate(tfr_writers):
                    if lod:
                        img = img.astype(np.float32)
                        img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
                    quant = np.rint(img).clip(0, 255).astype(np.uint8)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
                    tfr_writer.write(example.SerializeToString())
            except:
                print(sys.exc_info()[1])

    glob_pattern = os.path.join('datasets', tfrecord_dir, 'reals_all', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    images_num = len(image_filenames)
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(10000):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), img)

def create_lsun64_tfr(data_dir, tfrecord_dir='lsun64'):
    # "data_dir" should be a folder containing mdb files

    resolution_log2 = int(np.log2(64))

    # reading raw images
    import lmdb
    import cv2
    import io
    import sys
    import PIL.Image

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals_all'))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/lsun64%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    with lmdb.open(data_dir, readonly=True).begin(write=False) as txn:

        count = 0

        for idx, (key, value) in enumerate(txn.cursor()):
            try:
                try:
                    img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                    if img is None:
                        raise IOError('cv2.imdecode failed')
                    img = img[:, :, ::-1] # BGR => RGB
                except IOError:
                    img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                crop = np.min(img.shape[:2])
                img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                img = PIL.Image.fromarray(img, 'RGB')
                img = img.resize((64, 64), PIL.Image.ANTIALIAS)
                img = np.asarray(img)
                imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals_all', '%06d.png' % count), img)
                count += 1
                for lod, tfr_writer in enumerate(tfr_writers):
                    if lod:
                        img = img.astype(np.float32)
                        img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
                    quant = np.rint(img).clip(0, 255).astype(np.uint8)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                        'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
                    tfr_writer.write(example.SerializeToString())
            except:
                print(sys.exc_info()[1])

    glob_pattern = os.path.join('datasets', tfrecord_dir, 'reals_all', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    images_num = len(image_filenames)
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(10000):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
        imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), img)

def create_portrait_tfr(data_dir, tfrecord_dir='portrait'):
    # "data_dir" should be a folder containing jpg files

    resolution_log2 = int(np.log2(256))

    # reading raw images
    import PIL.Image
    glob_pattern = os.path.join(data_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))
    images_num = len(image_filenames)

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/portrait%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]).resize((256,256)))
        if idx < 10000:
            imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), img)
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())


def create_cartoon_tfr(data_dir, tfrecord_dir='cartoon'):
    # "data_dir" should be a folder containing 202599 218*178 png files

    resolution_log2 = int(np.log2(96))

    # reading raw images
    import PIL.Image
    glob_pattern = os.path.join(data_dir, '*.jpg')
    image_filenames = sorted(glob.glob(glob_pattern))

    # creating tfrecords
    if not os.path.exists('datasets'):
        os.mkdir('datasets')
    assert os.path.exists(os.path.join('datasets', tfrecord_dir))==False
    os.mkdir(os.path.join('datasets', tfrecord_dir))
    os.mkdir(os.path.join('datasets', tfrecord_dir, 'reals'))

    tfr_writers = []
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)

    for lod in range(resolution_log2 - 1):
        tfr_file = os.path.join('datasets', tfrecord_dir) + '/cartoon%02d.tfrecords' % (resolution_log2 - lod)
        tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))

    images_num = len(image_filenames)
    order = np.arange(images_num)
    np.random.RandomState(123).shuffle(order)

    for idx in range(images_num):
        img = np.asarray(PIL.Image.open(image_filenames[order[idx]]).resize((96,96)))
        if idx < 10000:
            imageio.imsave(os.path.join('datasets', tfrecord_dir, 'reals', '%06d.png' % idx), img)
        for lod, tfr_writer in enumerate(tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[0::2, 0::2] + img[0::2, 1::2] + img[1::2, 0::2] + img[1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            example = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(example.SerializeToString())

def data_iterator(dataset, lod_in, batch_size, resolution_log2):
    
    tfrecord_dir = os.path.join('datasets', dataset)
    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    images = []
    count = 0

    while True:
        for i, record in enumerate(tf.python_io.tf_record_iterator(os.path.join(tfrecord_dir, '%s%02d.tfrecords' % (dataset, int(resolution_log2-np.floor(lod_in)))), tfr_opt)):
            count += 1
            images.append(parse_tfrecord_np(record))
            if count >= batch_size:
                yield np.asarray(images)
                count = 0
                images = []

if __name__ == "__main__":

    tf.app.flags.DEFINE_string('dataset', 'mnist', 'Dataset to use')
    tf.app.flags.DEFINE_string('path', None, 'raw image path')
    FLAGS = tf.app.flags.FLAGS

    exec('create_%s_tfr(data_dir=FLAGS.path)' % FLAGS.dataset)
