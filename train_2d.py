import tensorflow as tf
import numpy as np
import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

tf.app.flags.DEFINE_string('gpu', '0', 'GPU(s) to use')
tf.app.flags.DEFINE_string('dataset', 'mixture_of_gaussians', 'Data distribution to use')
tf.app.flags.DEFINE_string('divergence', 'KL', 'Divergence')
tf.app.flags.DEFINE_string('path', './results', 'Output path')
tf.app.flags.DEFINE_integer('seed', 123, 'Random Seed')
tf.app.flags.DEFINE_integer('num_steps', 200000, 'Number of steps for training (outer loop)')
tf.app.flags.DEFINE_integer('batch_size', 1000, 'Batch size')
tf.app.flags.DEFINE_integer('plot_size', 20000, 'Number of points for plot')
tf.app.flags.DEFINE_integer('pool_size', 1, 'Number of batches of a pool')
tf.app.flags.DEFINE_integer('T', 1, 'Number of loops for moving particles (inner loop)')
tf.app.flags.DEFINE_float('step_size', 1.0, 'Step size for moving particles')
tf.app.flags.DEFINE_integer('U', 1, 'Number of loops for training D')
tf.app.flags.DEFINE_integer('L', 1, 'Number of loops for training G')

tf.app.flags.DEFINE_float('coef_smoothing', 0.99, 'Coefficient of generator moving average')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

np.random.seed(FLAGS.seed)
tf.random.set_random_seed(FLAGS.seed)

out_path = os.path.join(FLAGS.path, '%s-%s' % (FLAGS.dataset, FLAGS.divergence))
if not os.path.exists(out_path):
    os.makedirs(out_path)

def coef_div(d_score, coef=FLAGS.step_size):

    if FLAGS.divergence == 'KL':
        s = np.ones_like(d_score)
    elif FLAGS.divergence == 'LogD':
        s = 1 / (1 + np.exp(d_score))
    elif FLAGS.divergence == 'JS': # ensure numerical stablity
        s = 1 / (1 + 1/(1e-6 + np.exp(d_score)))
    elif FLAGS.divergence == 'Jef': # ensure numerical stablity
        s = np.clip(1 + np.exp(d_score), 1, 100)

    return coef * np.reshape(s, [-1, 1])

def data_generator(size=FLAGS.batch_size):

    if FLAGS.dataset == 'mixture_of_gaussians':
        theta = np.random.randint(8, size=size) / 8
        x = np.array([10 * np.cos(2 * np.pi * theta), 10 * np.sin(2 * np.pi * theta)]).T + np.random.randn(size, 2)
    elif FLAGS.dataset == 'concentric_circles':
        theta1 = np.random.uniform(0, 1, size)
        x1 = np.array([15*np.cos(2 * np.pi * theta1), 15*np.sin(2 * np.pi * theta1)]).T + np.random.randn(size, 2)*0.5
        theta2 = np.random.uniform(0, 1, size)
        x2 = np.array([5*np.cos(2 * np.pi * theta2), 5*np.sin(2 * np.pi * theta2)]).T + np.random.randn(size, 2)*0.5
        B = np.random.binomial(1, 0.67, size)
        M = np.stack((B, B), axis=1)
        x = M*x1 + (1-M)*x2

    return x

def plot(x, filename):

    x_1 = x[:, 0]
    x_2 = x[:, 1]
    plt.figure()
    sns_plot = sns.jointplot(x_1, x_2, kind='kde', color="skyblue", space=0, xlim=(-20, 20), ylim=(-20, 20))
    plt.savefig(os.path.join(out_path, '%s.png' % filename), dpi=300, quality=95)
    plt.close('all')

def distance(p1, p2):

    p1_expand = np.expand_dims(p1, axis=0)
    p2_expand = np.expand_dims(p2, axis=1)
    distance_matrix = np.sqrt(np.sum((p1_expand - p2_expand)**2, axis=2))
    Chamfer_distance = np.mean(np.min(distance_matrix, axis=1) + np.min(distance_matrix, axis=0))

    return Chamfer_distance

def generator(latents_in, is_smoothing=False, reuse=None):

    scope = 'generator_smoothing' if is_smoothing else 'generator'
    with tf.variable_scope(scope, reuse=reuse):
        h = tf.layers.dense(latents_in, units=256, activation=tf.nn.relu)
        samples_out = tf.layers.dense(h, units=2)

        return samples_out

def discriminator(samples_in, reuse=None):

    with tf.variable_scope('discriminator', reuse=reuse):
        h = tf.layers.dense(samples_in, units=256, activation=tf.nn.relu)
        scores_out = tf.layers.dense(h, units=1)

        return scores_out

################################ building graph ################################

x_p    = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='real_data')
z_p    = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='batch_latents')
Sz_p   = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='pool_latents')
G_z_p  = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='batch_particles')
G_Sz_p = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='pool_particles')

G_z  = generator(z_p)
Gs_z = generator(z_p, is_smoothing = True)

# discriminator loss:
d_real_logits = discriminator(x_p)
d_fake_logits = discriminator(G_z_p, reuse=True)
loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits)))
loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)))
loss_d = loss_d_real + loss_d_fake

# computing gradient:
d_fake_grad = tf.gradients(d_fake_logits, G_z_p)[0]

# generator loss:
loss_g = 0.5 * tf.reduce_mean(tf.reduce_sum((G_z - G_Sz_p)**2))

# optimizers:
vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

optimizer_d = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_d')
update_d    = optimizer_d.minimize(loss_d, var_list=vars_d)

optimizer_g = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.0, beta2=0.99, epsilon=1e-8, name='opt_g')
update_g    = optimizer_g.minimize(loss_g, var_list=vars_g)

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

x = data_generator(FLAGS.plot_size)
plot(x, 'ground_truth')

step_plot = []
log_chamfer_distance = []

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(FLAGS.num_steps+1):

        # sample a latent pool and get particles:
        Sz = np.random.uniform(-1, 1, (FLAGS.batch_size*FLAGS.pool_size, 2))
        P = sess.run(G_z, feed_dict={z_p: Sz})

        # inner loop:
        for t in range(FLAGS.T):

            # optimize discriminator:
            for u in range(FLAGS.U):

                # get a batch of real samples:
                x = data_generator()

                # sample a batch of latents from the pool:
                sample_index = np.random.choice(FLAGS.batch_size*FLAGS.pool_size, FLAGS.batch_size, replace=False)

                # update
                sess.run([update_d, update_gs], feed_dict={x_p: x, G_z_p: P[sample_index]})

            # move particles
            d_score = sess.run(d_fake_logits, feed_dict={G_z_p: P})
            grad = sess.run(d_fake_grad, feed_dict={G_z_p: P})
            P += coef_div(d_score) * grad

        # optimize generator:
        for l in range(FLAGS.L):

            sess.run(update_g, feed_dict={z_p: Sz, G_Sz_p: P})

        if not step % 100:

            real_loss, fake_loss = sess.run([loss_d_real, loss_d_fake], feed_dict={x_p: x, G_z_p: P[sample_index]})
            G_loss = sess.run(loss_g, feed_dict={z_p: Sz, G_Sz_p: P})
            print('Step: %d ' % step, 'D real loss: %.6f' % real_loss, '  |  D fake loss: %.6f' % fake_loss, '  |  Projection loss: %.6f' % G_loss)
            generated_samples = sess.run(Gs_z, feed_dict={z_p: np.random.uniform(-1, 1, (FLAGS.batch_size, 2))})
            cd = distance(x, generated_samples)
            step_plot.append(step)
            log_chamfer_distance.append(np.log(cd))

            if not step % 2000:
                generated_samples = sess.run(Gs_z, feed_dict={z_p: np.random.uniform(-1, 1, (FLAGS.plot_size, 2))})
                plot(generated_samples, 'fakes%d' % step)

    plt.figure()
    plt.style.use('ggplot')
    plt.plot(step_plot, log_chamfer_distance, label='VGrow-%s' % FLAGS.divergence)
    plt.xlabel("Training steps")
    plt.ylabel("log Chamfer distance")
    plt.legend()
    plt.savefig(os.path.join(out_path, 'plot.png'), dpi=300, quality=95)
    plt.close('all')

    f = open(os.path.join(out_path, 'log_chamfer-%s-%s.txt' % (FLAGS.dataset, FLAGS.divergence)),'w')
    f.write(str(step_plot))
    f.write('\n')
    f.write(str(log_chamfer_distance))
    f.close()
