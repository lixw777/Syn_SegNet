import tensorflow as tf
import numpy as np
import copy
import ipdb
import skimage.measure

from six.moves import xrange
def get_mean(matrix):
    result = list()
    for i in matrix:
        if not i.size:
            continue
        else:
            result.append(sum(i)/len(i))
    return np.array(result)

def get_psnr_mean(matrix, size):
    result = list()
    for i in matrix:
        if not i.size:
            continue
        else:
            # result.append(np.mean(i))
            result.append(np.sum(i)/size**2)
    return np.array(result)

def generate_mask(temp_pet, output):
    mask = np.ma.masked_where(temp_pet == -1, temp_pet)
    mask = np.ma.getmask(mask)
    masked_temp_pet = temp_pet
    masked_output = copy.deepcopy(output)
    masked_output[mask] = -1
    mask = np.ma.masked_where(temp_pet != -1, temp_pet)
    mask = np.ma.getmask(mask)
    return mask, masked_temp_pet, masked_output

def cosine_decay(learning_rate, global_step, decay_steps, alpha=0):
    # global_step = global_step % decay_steps
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    decayed_learning_rate = learning_rate * decayed

    return decayed_learning_rate



def get_shape(tensor):
    return tensor.get_shape().as_list()

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def batch_norm(*args, **kwargs):
    with tf.name_scope('bn'):
        bn = tf.layers.batch_normalization(*args, **kwargs)
    return bn

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)

def _l2normalize(v, eps=1e-12):
    return v / (tf.reduce_sum(v**2)**0.5 + eps)

def spectral_normed_weight(weights, num_iters=1, update_collection=None, with_sigma=False):
    w_shape = get_shape(weights)
    w_mat = tf.reshape(weights, [-1, w_shape[-1]]) # convert to 2 dimension but total dimension still the same
    u = tf.get_variable('u', [1, w_shape[-1]],
                        initializer=tf.truncated_normal_initializer(),
                        trainable=False)
    u_ = u
    for _ in range(num_iters):
        v_ = _l2normalize(tf.matmul(u_, w_mat, transpose_b=True))
        u_ = _l2normalize(tf.matmul(v_, w_mat))
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_, w_mat), u_, transpose_b=True))
    w_mat /=sigma

    if update_collection is None:
        with tf.control_dependencies([u.assign(u_)]):
            w_bar = tf.reshape(w_mat, w_shape)
    else:
        w_bar = tf.reshape(w_mat, w_shape)
        if update_collection != 'NO_OPS':
            tf.add_to_collection(update_collection, u.assign(u_))
    if with_sigma:
        return w_bar, sigma
    else:
        return w_bar

def snconv3d(input_, output_dim, size, stride, sn_iters=1, update_collection=None, filter_name='filter'):
    w = tf.get_variable(filter_name, [size, size, size, get_shape(input_)[-1], output_dim])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME')
    return conv

def snconv3d_tranpose(input_, output_dim_from, size, stride, sn_iters=1, update_collection=None):
    w = tf.get_variable('filter', [size, size, size, get_shape(output_dim_from)[-1], get_shape(input_)[-1]])
    w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
    conv = tf.nn.conv3d_transpose(input_, w_bar, strides=[1, stride, stride, stride, 1], padding='SAME',
                                  output_shape=tf.shape(output_dim_from))
    return conv

def snconv3d_1x1(input_, output_dim, sn_iters=1, sn=True, update_collection=None, init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
    with tf.variable_scope(name):
        w = tf.get_variable('filter', [1, 1, 1, get_shape(input_)[-1], output_dim], initializer=init)
        if sn:
            w_bar = spectral_normed_weight(w, num_iters=sn_iters, update_collection=update_collection)
        else:
            w_bar = w
        conv = tf.nn.conv3d(input_, w_bar, strides=[1, 1, 1, 1, 1], padding='SAME')
        return conv


def sn_attention(name, x, sn=True, final_layer=False, update_collection=None, as_loss=False):
    with tf.variable_scope(name):
        batch_size, height, width, depth, num_channels = x.get_shape().as_list()
        if not batch_size:
            batch_size = 1
        location_num = height * width * depth

        if final_layer:
            downsampled = location_num//(64**3)#// 整除，返回商的整数部分   ** 求幂/次方
            stride = 64
        else:
            downsampled = location_num // 8
            stride = 2
        #ipdb.set_trace()
        # theta path
        theta = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//8 or 4), update_collection=update_collection, name='theta')#(?, 8, 8, 8, 64)
        theta = tf.reshape(theta, [batch_size, location_num, (num_channels//8 or 4)])#shape=(1, 512, 64)

        # phi path
        phi = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//8 or 4), update_collection=update_collection, name='phi')#(?, 8, 8, 8, 64)
        phi = tf.layers.max_pooling3d(inputs=phi, pool_size=[2, 2, 2], strides=stride)
        phi = tf.reshape(phi, [batch_size, downsampled, (num_channels//8 or 4)])#shape=(1, 512, 64)

        attn = tf.matmul(theta, phi, transpose_b=True)#(1, 512, 512)
        attn = tf.nn.softmax(attn)

        # g path
        g = snconv3d_1x1(x, sn=sn, output_dim=(num_channels//2 or 16), update_collection=update_collection, name='g')#shape=(?, 8, 8, 8, 256)
        g = tf.layers.max_pooling3d(inputs=g, pool_size=[2, 2, 2], strides=stride)
        g = tf.reshape(g, [batch_size, downsampled, (num_channels//2 or 16)])#shape=(1, 512, 256)

        attn_g = tf.matmul(attn, g)#shape=(1, 512, 256)
        attn_g = tf.reshape(attn_g, [batch_size, height, width, depth, (num_channels//2 or 16)])#(1, 8, 8, 8, 256)
        sigma = tf.get_variable('sigma_ratio', [], initializer=tf.constant_initializer(0.0), trainable=True)
        attn_g = snconv3d_1x1(attn_g, sn=sn, output_dim=num_channels, update_collection=update_collection, name='attn')
        if as_loss:
            return attn_g
        else:
            return (x + sigma * attn_g)/(1+sigma), sigma

def cross_entroy_loss(logits, labels):
    #ipdb.set_trace()
    #pred 未经softmax的网络最后一层输出  y真实标签的onehot

    labels = tf.reshape(labels, (-1, 8))
    logits = tf.reshape(logits, (-1, 8))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def weighted_ce_loss(logits, labels):
    labels = tf.dtypes.cast(labels, tf.float32)
    labels = tf.reshape(labels, (-1, 8))
    logits = tf.reshape(logits, (-1, 8))
    class_weights = tf.constant([[1.0, 2.5, 4.0, 3.0, 3.0, 3.5, 4.5, 5.0]])
    weights = tf.reduce_mean(class_weights * labels, axis=1)
    cross1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    weighted_losses = cross1 * weights
    return tf.reduce_mean(weighted_losses)

def WACE_loss(logits, labels, label1_mask, label2_mask,label3_mask,label4_mask,label5_mask,label6_mask,label7_mask):
    ce_loss1 = tf.reduce_mean(label1_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss2 = tf.reduce_mean(label2_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss3 = tf.reduce_mean(label3_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss4 = tf.reduce_mean(label4_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss5 = tf.reduce_mean(label5_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss6 = tf.reduce_mean(label6_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss7 = tf.reduce_mean(label7_mask * (0.8 * weighted_ce_loss(logits, labels)))
    ce_loss = ce_loss1 + ce_loss2 + ce_loss3 + ce_loss4 + ce_loss5 + ce_loss6 + ce_loss7
    return ce_loss

def categorical_focal_loss(Y_pred, Y_gt):
    """
     Categorical focal_loss between an output and a target
    :param Y_pred: A tensor of the same shape as `y_pred`
    :param Y_gt: A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param alpha: Sample category weight,which is shape (C,) where C is the number of classes
    :param gamma: Difficult sample weight
    :return:
    """

    Y_gt = tf.dtypes.cast(Y_gt, tf.float32)
    weight_loss = np.array([1.0, 2.5, 4.0, 3.0, 3.0, 3.5, 4.5, 5.0])
    epsilon = 1.e-5
    gamma = 2
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weight_loss * loss)
    return loss

def dice_coef_loss(Y_pred, Y_gt):
    # ipdb.set_trace()
    #Y_pred after softmax, Y_gt onehot
    Y_gt = tf.dtypes.cast(Y_gt, tf.float32)
    Z, H, W, C = Y_gt.get_shape().as_list()[1:]
    smooth = 1e-5
    pred_flat = tf.reshape(Y_pred, [-1, H * W * C * Z])
    true_flat = tf.reshape(Y_gt, [-1, H * W * C * Z])
    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
    denominator = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) + smooth
    loss = 1 - tf.reduce_mean(intersection / denominator)
    return loss

def get_whole_hippo_mask(data):
    return data > 0
def get_1_mask(data):
    return data == 1
def get_2_mask(data):
    return data == 2
def get_3_mask(data):
    return data == 3
def get_4_mask(data):
    return data == 4
def get_5_mask(data):
    return data == 5
def get_6_mask(data):
    return data == 6
def get_7_mask(data):
    return data == 7
def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

def dice_eval(pred, gt):
    header = ("WholeHippo", "SUB", "CA2", "CA1", "CA4-DG", "ERC", "CA3", "Tail")
    masking_functions = (
    get_whole_hippo_mask, get_1_mask, get_2_mask, get_3_mask, get_4_mask, get_5_mask, get_6_mask, get_7_mask)
    dice = []
    truth = gt
    prediction = pred
    dice.append([dice_coefficient(func(truth), func(prediction)) for func in masking_functions])
    return dice

def squeeze_and_excitation(input_x, out_dim, ratio, layer_name):
    #ipdb.set_trace()
    with tf.name_scope(layer_name):
        squeeze = tf.keras.layers.GlobalAveragePooling3D()(input_x)#计算每一张特征图的所有像素点的均值
        excitation = tf.keras.layers.Dense(units=out_dim / ratio)(squeeze)
        excitation = tf.nn.relu(excitation)
        excitation = tf.keras.layers.Dense(units=out_dim)(excitation)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, 1, out_dim])
        scale = input_x * excitation
        return scale