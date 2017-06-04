import tensorflow as tf

def trafficNet(x,num_filters=16):
    """
    This network is a modified LeNet used for classifying traffic signals
    """
    mu = 0
    sigma = 0.1
    """
    First set of convolution and pooling layer
    input : 32x32x1
    output : 28x28xnum_filters
    """
    conv1_w = tf.Variable(tf.truncated_normal([5,5,1,num_filters],mean=mu,stddev=sigma))
    conv1_b = tf.Variable(tf.zeros([num_filters]))
    conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding='VALID')
    conv1 = tf.nn.bias_add(conv1,conv1_b) # convolution layer 1 # 28x28x16
    conv1_activ = tf.nn.relu(conv1) # activation after layer 1
    conv1_pool = tf.nn.max_pool(conv1_activ,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID') # 14x14x16

    """
    Second set of convolution and pooling layer
    input : 14x14xnum_filters
    output : 5x5x(2num_filters)
    """

    conv2_w = tf.Variable(tf.truncated_normal([5,5,num_filters,2*num_filters], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros([2*num_filters]))
    conv2 = tf.nn.conv2d(conv1_pool,conv2_w,strides=[1,1,1,1],padding='VALID')
    conv2 = tf.nn.bias_add(conv2,conv2_b) # convolution layer 2 # 10x10x32
    conv2_activ = tf.nn.relu(conv2) # activation layer 2
    conv2_pool = tf.nn.max_pool(conv2_activ,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') # 5x5x32

    layer_flatten = flatten(conv_pool2)
    layer_dropout = tf.nn.dropout(layer_flatten,keep_prob=0.8)

     # Fully Connected Layer
    layer_fc1 = tf.contrib.layers.fully_connected(layer_dropout, int(num_filters*8), tf.nn.relu)

    tf.contrib.layers.fully_connected
    # Fully Connected Layer
    layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, int(num_filters*4), tf.nn.relu)

    # Fully Connected Layer
    layer_fc3 = tf.contrib.layers.fully_connected(layer_fc2, 43, tf.nn.relu)

    logits = layer_fc3
    return logits

