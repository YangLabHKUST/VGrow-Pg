import tensorflow as tf
import numpy as np
import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

import data_tool
from networks import *
from utils import *

tf.app.flags.DEFINE_string('gpu', '0', 'GPU(s) to use')
tf.app.flags.DEFINE_string('dataset', 'mnist', 'Dataset to use')
tf.app.flags.DEFINE_string('divergence', 'KL', 'Divergence')
tf.app.flags.DEFINE_string('path', './results', 'Output path')
tf.app.flags.DEFINE_integer('seed', 1234, 'Random Seed')
tf.app.flags.DEFINE_integer('init_resolution', 4, 'Initial resolution of images')
tf.app.flags.DEFINE_integer('z_dim', 512, 'Size of latent vectors')
tf.app.flags.DEFINE_integer('dur_nimg', 600000, 'Number of images used for a phase')
tf.app.flags.DEFINE_integer('total_nimg', 18000000, 'Total number of images used for training')
tf.app.flags.DEFINE_integer('pool_size', 1, 'Number of batches of a pool')
tf.app.flags.DEFINE_integer('T', 1, 'Number of loops for moving particles')
tf.app.flags.DEFINE_float('step_size', 1.0, 'Step size for moving particles')
tf.app.flags.DEFINE_integer('U', 1, 'Number of loops for training D')
tf.app.flags.DEFINE_integer('L', 1, 'Number of loops for training G')
tf.app.flags.DEFINE_integer('num_row', 10, 'Number images in a line of image grid')
tf.app.flags.DEFINE_integer('num_line', 10, 'Number images in a row of image grid')

tf.app.flags.DEFINE_bool('use_gp', True, 'Use gradient penalty?')
tf.app.flags.DEFINE_float('coef_gp', 1., 'Coefficient of gradient penalty')
tf.app.flags.DEFINE_float('target_gp', 1., 'Target of gradient penalty')

tf.app.flags.DEFINE_float('coef_smoothing', 0.99, 'Coefficient of generator moving average')

tf.app.flags.DEFINE_bool('resume_training', False, 'Resume Training?')
tf.app.flags.DEFINE_integer('resume_num', 0, 'Resume number of images')

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
if not os.path.exists(out_path):
    os.makedirs(out_path)

batchsize_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}

def lod(num_img):

    ph_num = num_img // (2*FLAGS.dur_nimg)
    remain_num = num_img - ph_num * (2*FLAGS.dur_nimg)

    if np.log2(resolution/FLAGS.init_resolution) <= ph_num:
        return 0.
    elif remain_num <= FLAGS.dur_nimg:
        return np.log2(resolution/FLAGS.init_resolution) - ph_num
    else: 
        return np.log2(resolution/FLAGS.init_resolution) - ph_num - (remain_num - FLAGS.dur_nimg) / FLAGS.dur_nimg

def coef_div(d_score, coef=FLAGS.step_size):

    if FLAGS.divergence == 'KL':
        s = np.ones_like(d_score)
    elif FLAGS.divergence == 'LogD':
        s = 1 / (1 + np.exp(d_score))
    elif FLAGS.divergence == 'JS': # ensure numerical stablity
        s = 1 / (1 + 1/(1e-6 + np.exp(d_score)))
    elif FLAGS.divergence == 'Jef': # ensure numerical stablity
        s = np.clip(1 + np.exp(d_score), 1, 100)

    return coef * np.reshape(s, [-1, 1, 1, 1])

################################ building graph ################################

x_p    = tf.placeholder(dtype=tf.float32, shape=[None, resolution, resolution, num_channels], name='images')
z_p    = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.z_dim], name='batch_latents')
Sz_p   = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.z_dim], name='pool_latents')
G_z_p  = tf.placeholder(dtype=tf.float32, shape=[None, resolution, resolution, num_channels], name='batch_particles')
G_Sz_p = tf.placeholder(dtype=tf.float32, shape=[None, resolution, resolution, num_channels], name='pool_particles')
lod_in = tf.placeholder(dtype=tf.float32, shape=[], name='level_of_details')

G_z  = generator(z_p, lod_in, num_channels, resolution, FLAGS.z_dim, num_features)
Gs_z = generator(z_p, lod_in, num_channels, resolution, FLAGS.z_dim, num_features, is_smoothing = True)

# discriminator loss:
d_real_logits = discriminator(x_p, lod_in, num_channels, resolution, num_features)
d_fake_logits = discriminator(G_z_p, lod_in, num_channels, resolution, num_features, reuse=True)
loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
loss_d = loss_d_real + loss_d_fake

# computing gradient:
d_fake_grad = tf.gradients(d_fake_logits, G_z_p)[0]

# if use gradient penalty:
if FLAGS.use_gp:
    mix_factors = tf.placeholder(dtype=tf.float32, shape=[None, 1, 1, 1], name='mix_factors')
    mix_images = x_p + mix_factors * (G_z_p - x_p)
    d_mix_logits = discriminator(mix_images, lod_in, num_channels, resolution, num_features, reuse=True)
    d_mix_grad = tf.gradients(d_mix_logits, mix_images)[0]
    mix_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(d_mix_grad), axis=[1,2,3]))
    mix_grad_penalty = tf.square(mix_grad_norm - FLAGS.target_gp)
    loss_d += 0.5 * FLAGS.coef_gp * tf.reduce_mean(mix_grad_penalty)

# generator loss:
loss_g = 0.5 * tf.reduce_mean(tf.reduce_sum((G_z - G_Sz_p)**2))

# optimizers:
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

optimizer_d = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_d')
update_d = optimizer_d.minimize(loss_d, var_list=vars_d)

optimizer_g = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_g')
update_g = optimizer_g.minimize(loss_g, var_list=vars_g)

reset_optimizer_d = tf.variables_initializer(optimizer_d.variables())
reset_optimizer_g = tf.variables_initializer(optimizer_g.variables())

################################ moving average of generator ################################

G_trainables  = OrderedDict([(var.name[len('generator/'):], var) for var in tf.trainable_variables('generator' + '/')])
Gs_trainables = OrderedDict([(var.name[len('generator_smoothing/'):], var) for var in tf.trainable_variables('generator_smoothing' + '/')])

with tf.name_scope('generator_smoothing/'):
    with tf.name_scope('smoothing'):
        ops = []
        for name, var in Gs_trainables.items():
            new_value = G_trainables[name] + (var - G_trainables[name]) * FLAGS.coef_smoothing
            ops.append(var.assign(new_value))
        update_gs = tf.group(*ops)

################################ training ################################

saver = tf.train.Saver()

resolution_log2 = int(np.log2(resolution))

with tf.Session() as sess:

    iterators = [data_tool.data_iterator(dataset=FLAGS.dataset, lod_in=lod, batch_size=batchsize_dict[2**(resolution_log2-lod)], resolution_log2=resolution_log2) for lod in range(int(np.log2(resolution/FLAGS.init_resolution))+1)]

    if not FLAGS.resume_training:
        sess.run(tf.global_variables_initializer())
        num_img = 0
        tick_kimg = 0
        prev_lod = -1.0

    else:
        saver.restore(sess, os.path.join(out_path, 'networks-%08d.ckpt' % FLAGS.resume_num))
        num_img = FLAGS.resume_num
        tick_kimg = (num_img // 1000)
        
    cur_lod = lod(num_img)
    z_fixed = np.random.randn(10000, FLAGS.z_dim)
    count = 0

    while num_img <= FLAGS.total_nimg:

        prev_lod = cur_lod

        # get mini-batch size:
        batch_size = batchsize_dict[2**int(resolution_log2-np.floor(cur_lod))]

        # sample a latent pool and get particles:
        Sz = np.random.randn(batch_size*FLAGS.pool_size, FLAGS.z_dim)
        P = sess.run(G_z, feed_dict={z_p: Sz, lod_in: cur_lod})

        # inner loop:
        for t in range(FLAGS.T):

            # optimize discriminator:
            for u in range(FLAGS.U):

                # get a batch of real images:
                x = next(iterators[int(np.floor(cur_lod))])
                x = process_real(x, cur_lod)
                num_img += batch_size

                # sample a batch of latents from the pool:
                sample_index = np.random.choice(batch_size*FLAGS.pool_size, batch_size, replace=False)

                # update
                if FLAGS.use_gp:
                    mix_coef = np.random.uniform(0, 1, [batch_size, 1, 1, 1])
                    sess.run([update_d, update_gs], feed_dict={x_p: x, G_z_p: P[sample_index], mix_factors: mix_coef, lod_in: cur_lod})
                else:
                    sess.run([update_d, update_gs], feed_dict={x_p: x, G_z_p: P[sample_index], lod_in: cur_lod})

            # move particles
            d_score = sess.run(d_fake_logits, feed_dict={G_z_p: P, lod_in: cur_lod})
            grad = sess.run(d_fake_grad, feed_dict={G_z_p: P, lod_in: cur_lod})
            P += coef_div(d_score) * grad

        # optimize generator:
        for l in range(FLAGS.L):

            sess.run(update_g, feed_dict={z_p: Sz, G_Sz_p: P, lod_in: cur_lod})

        cur_lod = lod(num_img)

        # reset Adam optimizers states when increasing resolution:
        if np.floor(cur_lod) != np.floor(prev_lod) or np.ceil(cur_lod) != np.ceil(prev_lod):
            sess.run([reset_optimizer_d, reset_optimizer_g])

        if (num_img // 1000) >= tick_kimg + 150:
            count += 1

            tick_kimg = (num_img // 1000)
            real_loss, fake_loss = sess.run([loss_d_real, loss_d_fake], feed_dict={x_p: x, G_z_p: P[sample_index], lod_in: cur_lod})
            G_loss = sess.run(loss_g, feed_dict={z_p: Sz, G_Sz_p: P, lod_in: cur_lod})
            
            print('num_img: %d ' % num_img, '  |  lod_in: %.2f' % cur_lod, '\n',
                  'D real loss: %.6f' % real_loss, '  |  D fake loss: %.6f' % fake_loss, '  |  Projection loss: %.6f' % G_loss)

            # generate an image grid
            gen_imgs = []
            for i in tqdm(range(FLAGS.num_row*FLAGS.num_line)):
                img = sess.run(Gs_z, feed_dict={z_p: np.expand_dims(z_fixed[i], axis=0), lod_in: cur_lod})
                gen_imgs.append(img[0])
            gen_imgs = np.array(gen_imgs)
            gen_imgs = (gen_imgs + 1) / 2
            imageio.imsave(os.path.join(out_path, 'fakes%06d.png' % (num_img // 1000)), montage(gen_imgs, grid=[FLAGS.num_row, FLAGS.num_line]))

            # generate images for FID
            if count == 4:
                count = 0

                if not os.path.exists(os.path.join(out_path, 'fakes%06d' % (num_img // 1000))):
                    os.makedirs(os.path.join(out_path, 'fakes%06d' % (num_img // 1000)))
                    
                for i in tqdm(range(10000)):
                    img = sess.run(Gs_z, feed_dict={z_p: np.expand_dims(z_fixed[i], axis=0), lod_in: cur_lod})
                    img = (img + 1) / 2
                    imageio.imsave(os.path.join(out_path, 'fakes%06d' % (num_img // 1000), '%05d.png' % i), np.rint(img[0]*255).clip(0, 255).astype(np.uint8))

                # save model
                saver.save(sess, os.path.join(out_path, 'networks-%08d.ckpt' % num_img))

