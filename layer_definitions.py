import tensorflow as tf

def conv_conv_pool(input_, output_filt, input_filt,training, name, pool=True, activation=tf.nn.relu):
    
     net = input_
     with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(output_filt):
            #Define the Convolution layer
            """
            conv2d(inputs,filters,kernel_size,strides=(1, 1),padding='valid',
                   data_format='channels_last',
                   dilation_rate=(1, 1),
                   activation=None,
                   use_bias=True,
                   kernel_initializer=None,
                   bias_initializer=tf.zeros_initializer(),
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   kernel_constraint=None,
                   bias_constraint=None,
                   trainable=True,
                   name=None,
                   reuse=None
                   )
            """
            #Initializer for kernel weights using a Normal Distribution
            ini = tf.contrib.layers.xavier_initializer(uniform = False,seed = None,dtype = tf.float32)
            
            net = tf.layers.conv2d(net, F, (3, 3),strides = (1,1), activation=None, padding='valid',
                                   kernel_initializer = ini, name="conv_{}".format(i + 1), reuse = tf.AUTO_REUSE)
            
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net
        #2x2 Maxpool with stride 2
        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA_smaller, inputB_larger, name,size,filters):
       """Upsample `inputA` and concat with `input_B`
       Args:
           input_A (4-D Tensor): (N, H, W, C)
           input_B (4-D Tensor): (N, 2*H, 2*H, C2)
           name (str): name of the concat operation
           Returns:
           output (4-D Tensor): (N, 2*H, 2*W, C + C2)
       """
       inputA = inputA_smaller
       inputB = inputB_larger
       H_a, W_a, _ = inputA.get_shape().as_list()[1:]
       H_b, W_b, _ = inputB.get_shape().as_list()[1:]
       
       a, b = size
       target_H = H_a * a
       target_W = W_a * b
       begin = [0,int((H_b-target_H)/2),int((W_b-target_W)/2),0]
       size_slice = [-1,target_H,target_W,-1]
       sliced_inputB = tf.slice(inputB,begin,size_slice,name = "slice_{}".format(name))
       
       upsampled_inputB = tf.layers.conv2d_transpose(inputA,filters,[2,2],strides = (2,2), padding = 'valid')
       
       return tf.concat([sliced_inputB,upsampled_inputB],axis = -1,name = "concat_{}".format(name))