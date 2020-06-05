import tensorflow as tf
import numpy as np
import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from tqdm import tqdm

import data_tool
from networks import *

tf.app.flags.DEFINE_string('gpu', '0', 'GPU(s) to use')
tf.app.flags.DEFINE_string('dataset', 'mnist', 'Dataset to use')
tf.app.flags.DEFINE_string('divergence', 'KL', 'Divergence')
tf.app.flags.DEFINE_string('path', './results', 'Output path')
tf.app.flags.DEFINE_string('dir_name', 'fakes', 'Folder to save generated images')
tf.app.flags.DEFINE_integer('seed', 123, 'Random Seed')
tf.app.flags.DEFINE_integer('init_resolution', 4, 'Initial resolution of images')
tf.app.flags.DEFINE_integer('z_dim', 512, 'Size of latent vectors')
tf.app.flags.DEFINE_integer('dur_nimg', 600000, 'Number of images used for a phase')

tf.app.flags.DEFINE_integer('load_num', 0, 'Generate fake images using networks trained with given number of real images')

tf.app.flags.DEFINE_integer('generate_num', 10000, 'The number of images you want to generate')

tf.app.flags.DEFINE_bool('interpolation', False, 'Generate interpolation results?')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

np.random.seed(FLAGS.seed)
tf.random.set_random_seed(FLAGS.seed)

def inferenceResolution(tfrecord_dir):
    assert os.path.isdir(tfrecord_dir)
    tfr_files = sorted(glob(os.path.join(tfrecord_dir, '*.tfrecords')))
    assert len(tfr_files) >= 1
    tfr_shapes = []
    for tfr_file in tfr_files:
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shapes.append(data_tool.parse_tfrecord_np(record).shape)
            break
    max_shape = max(tfr_shapes, key=lambda shape: np.prod(shape))
    resolution = max_shape[0]
    assert max_shape[-1] in [1, 3]
    num_channels = max_shape[-1]

    if resolution <= 128:
        num_features = 256 
    else:
        num_features = 128
    return num_channels, resolution, num_features

tfrecord_dir = os.path.join("datasets", FLAGS.dataset)
num_channels, resolution, num_features = inferenceResolution(tfrecord_dir)

out_path = os.path.join(FLAGS.path, '%s-%s' % (FLAGS.dataset, FLAGS.divergence))

def lod(num_img):

    ph_num = num_img // (2*FLAGS.dur_nimg)
    remain_num = num_img - ph_num * (2*FLAGS.dur_nimg)

    if np.log2(resolution/FLAGS.init_resolution) <= ph_num:
        return 0.
    elif remain_num <= FLAGS.dur_nimg:
        return np.log2(resolution/FLAGS.init_resolution) - ph_num
    else: 
        return np.log2(resolution/FLAGS.init_resolution) - ph_num - (remain_num - FLAGS.dur_nimg) / FLAGS.dur_nimg

################################ building graph ################################

z_p    = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.z_dim], name='batch_latents')
lod_in = tf.placeholder(dtype=tf.float32, shape=[], name='level_of_details')

Gs_z = generator(z_p, lod_in, num_channels, resolution, FLAGS.z_dim, num_features, is_smoothing = True)

################################ generating ################################

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, os.path.join(out_path, 'networks-%08d.ckpt' % FLAGS.load_num))
    num_img = FLAGS.load_num
        
    cur_lod = lod(num_img)
    if FLAGS.interpolation:
        pair = np.random.randn(2, FLAGS.z_dim)
        z_fixed = np.linspace(pair[0], pair[1], FLAGS.generate_num)
    else:
        z_fixed = np.random.randn(FLAGS.generate_num, FLAGS.z_dim)

    os.mkdir(os.path.join(out_path, FLAGS.dir_name))

    for i in tqdm(range(FLAGS.generate_num)):
        img = sess.run(Gs_z, feed_dict={z_p: np.expand_dims(z_fixed[i], axis=0), lod_in: cur_lod})
        img = (img + 1) / 2
        imageio.imsave(os.path.join(out_path, FLAGS.dir_name, '%05d.png' % i), np.rint(img[0]*255).clip(0, 255).astype(np.uint8))
