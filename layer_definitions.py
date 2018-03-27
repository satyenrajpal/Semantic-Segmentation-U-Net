import tensorflow as tf
import numpy as np

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
            
            net = tf.layers.batch_normalization(net, training=training, name="bn_{}".format(i + 1),reuse=tf.AUTO_REUSE)
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net
        #2x2 Maxpool with stride 2
        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))

        return net, pool


def upsample_concat(inputA, inputB, name,size,filters):
       """Upsample `inputA` and concat with `input_B`
       Args:
           input_A (4-D Tensor): (N, H, W, C) (smaller)
           input_B (4-D Tensor): (N, 2*H, 2*H, C2) (larger)
           name (str): name of the concat operation
           size=(a,b): a along height,b along width
           filters: number of filters
           Returns:
           output (4-D Tensor): (N, 2*H, 2*W, C + C2)
       """
       H_a, W_a, _ = inputA.get_shape().as_list()[1:]
       H_b, W_b, _ = inputB.get_shape().as_list()[1:]
       
       a, b = size
       target_H = H_a * a
       target_W = W_a * b
       #For slicing middle portion of input_B
       begin = [0,int((H_b-target_H)/2),int((W_b-target_W)/2),0]
       size_slice = [-1,target_H,target_W,-1]
       sliced_inputB = tf.slice(inputB,begin,size_slice,name = "slice_{}".format(name))
       
       upsampled_inputB = tf.layers.conv2d_transpose(inputA,filters,[2,2],strides = (2,2), padding = 'valid')
       
       return tf.concat([sliced_inputB,upsampled_inputB],axis = -1,name = "concat_{}".format(name))
   
def convert_classes_to_color(predictions):
        
        """Colors based on number of classes:
            0 - White - Road
            1 - Blue - House
            2 - Cyan - Low Vegetation
            3 - Green - Vegetation
            4 - Yellow - Car
            5 - Red - Water
        """
        print("Number of images to convert are {}".format(len(predictions)))
        final=[]
        for i in range(len(predictions)):
            a = predictions[i]
            h,w = a.shape[0],a.shape[1]
            b = np.zeros([h,w,3])
            x=1
            for j in range(0,h):
                for k in range(0,w):
                    #OpenCV works with BGR Format
                    #MatplotLib works iwth RGB
                    if (a[j,k] == 0):
                        b[j,k,:] = 255/x              #White if Road
                    elif (a[j,k] == 1):
                        b[j,k,0] = 255/x              #Blue if house
                    elif (a[j,k] == 2):
                        b[j,k,1] = b[j,k,0] = 255/x   #Cyan if low Vegetation
                    elif (a[j,k] == 3):
                        b[j,k,1] = 255/x              #Green if Vegetation
                    elif (a[j,k] == 4):
                        b[j,k,2]=b[j,k,1] = 255/x     #Yellow if Car
                    else:
                        b[j,k,2] =  255/x             #Red if Water

            final.append(b)
        return np.asarray(final) 
    
def get_next_batch(imgs,masks,num):
    """Return the next images of batch_size in the indices for training"""
    idx = np.random.choice(len(imgs),size=num,replace=False)
    img_batch = imgs[idx]
    masks_batch = masks[idx]
    sparse_idx_len=np.where(masks_batch==-1)[0].size
 
    assert len(img_batch) == num == len(masks_batch)
   
    return img_batch.astype(np.float32),masks_batch.astype(np.int32),sparse_idx_len

def make_unet(_input, training,classes):
    """Build a U-Net architecture
    Args:
    _input (4-D Tensor): (N, H, W, C)
    training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
    Returns:
    output (4-D Tensor): (N, H', W', C')
    Output shape with :
        H' = H - 184
        W' = W - 184
        C' = NUM_CLASSES
    """
    
    #net = X / 127.5 - 1
            
    X = tf.layers.conv2d(_input, 3, (1, 1), strides = (1,1), padding = 'same',name="color_adjust",reuse = tf.AUTO_REUSE)
                               
    conv1, pool1 = conv_conv_pool(X, [64,64], 3,training, name=1)
    
    conv2, pool2 = conv_conv_pool(pool1, [128, 128],64, training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [256, 256], 128,training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [512, 512], 256,training, name=4)
    conv5 = conv_conv_pool(pool4, [1024, 1024], 512, training, pool=False,name=5)
    up6 = upsample_concat(conv5, conv4, name=6, size=(2,2), filters = 512)
    conv6 = conv_conv_pool(up6, [512, 512], 1024, training, name=6, pool=False)
    
    up7 = upsample_concat(conv6, conv3, name=7,size = (2,2), filters = 256)
    conv7 = conv_conv_pool(up7, [256, 256], 512, training, name=7, pool=False)
       
    up8 = upsample_concat(conv7, conv2, name=8,size = (2,2),filters = 128)
    conv8 = conv_conv_pool(up8, [128, 128], 256, training, name=8, pool=False)
    
    up9 = upsample_concat(conv8, conv1, name=9, size = (2,2), filters = 64)
    conv9 = conv_conv_pool(up9, [64, 64], 128, training, name=9, pool=False)
    print("Model definition complete..")
    
    return tf.layers.conv2d(conv9, classes, (3, 3), strides = (1,1), name='logits', padding='same')
        