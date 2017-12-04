import math
import numpy as np 
import tensorflow as tf

def msra_stddev(x, k_h, k_w): 
    return 1/math.sqrt(0.5*k_w*k_h*x.get_shape().as_list()[-1])

def mse_ignore_nans(preds, targets, **kwargs):
    #Computes mse, ignores targets which are NANs
    
    # replace nans in the target with corresponding preds, so that there is no gradient for those
    targets_nonan = tf.where(tf.is_nan(targets), preds, targets)
    return tf.reduce_mean(tf.square(targets_nonan - preds), **kwargs)

def conv2d(input_, output_dim, 
        k_h=3, k_w=3, d_h=2, d_w=2, msra_coeff=1,
        name="conv2d", reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=msra_coeff * msra_stddev(input_, k_h, k_w)))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        return tf.nn.bias_add(tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME'), b)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, name='linear', msra_coeff=1):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=msra_coeff * msra_stddev(input_, 1, 1)))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b
    
def conv_encoder(data, params, name, msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        if nl == len(params)-1:
            params['out_channels'] = 3
            
        layers.append(lrelu(conv2d(curr_inp, 16, k_h=3, k_w=3, d_h=2, d_w=2, name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers[-1]

def rpn_conv_encoder(data, params, name, msra_coeff=1):

    # Regular CNN
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        layers.append(lrelu(conv2d(curr_inp, 16, k_h=3, k_w=3, d_h=2, d_w=2, name=name + str(nl), msra_coeff=msra_coeff)))
        #layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))

    # Hook layers
    reuse=False
    hooks = []
    for l in layers:
        rpn_layer = lrelu(conv2d(l, 8, k_h=3, k_w=3, d_h=2, d_w=2, name=name + '_rpn', msra_coeff=msra_coeff, reuse=reuse))
        hooks.append(rpn_layer)
        reuse=True

    # Flattened


    # Max pool to same size
    resized_imgs = [hooks[0]]

    resize_shape = hooks[0].get_shape().as_list()[1:3]

    for i in range(1, len(hooks)):
        resized_img = tf.image.resize_images(hooks[i], resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_imgs.append(resized_img)

    # Flattened
    """
    flattened = []
    for hook in resized_imgs:
        flattened.append(tf.layers.flatten(hook))
    output = tf.concat(flattened, axis=-1)
    """
    # Dense connections
    prev_layers = []
    curr_layer = None
    for d in range(len(resized_imgs)):
        if not prev_layers:
            curr_layer = resized_imgs[d]
        else:
            curr_layer = tf.concat(prev_layers+[resized_imgs[d]], axis=-1)

        curr_layer = lrelu(conv2d(curr_layer, 8, k_h=3, k_w=3, d_h=1, d_w=1, name=name + '_dense_'+str(d), msra_coeff=msra_coeff, reuse=False))
        prev_layers.append(curr_layer)

    output = prev_layers[-1]

    # NMS
    # concatted = tf.concat(resized_imgs, axis=-1)
    # output = tf.reduce_max(concatted, axis=-1)

    return output

        
def fc_net(data, params, name, last_linear = False, return_layers = [-1], msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        
        if nl == len(params) - 1 and last_linear:
            layers.append(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff))
        else:
            layers.append(lrelu(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff)))
            
    if len(return_layers) == 1:
        return layers[return_layers[0]]
    else:
        return [layers[nl] for nl in return_layers]

def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])
