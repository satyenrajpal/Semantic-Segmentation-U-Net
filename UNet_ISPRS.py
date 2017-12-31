
import tensorflow as tf
import cv2
import Create_dataset as ct
import numpy as np
from layer_definitions import conv_conv_pool,upsample_concat
import os
import random

class Model_Unet:
    def __init__(self):
        """
        in_dir: Directory containing the Images and Labelled Masks in the following directory structure 
        (Directory names are within -- --)
        --Images-- (Contains 3 folders)
            -- Train -- (training Images)
            -- Test -- (Test Images)
            -- Validation -- (Validation Images)
        -- Labels_classes -- (Contains 3 folders)
            -- Train -- (Training Masks)
            -- Validation -- (Validation Masks)
        """
        self.dataset = ct.cache(cache_path='my_dataset_cache.pkl', fn=ct.Dataset,in_dir='vaihingen\data_labels')
        #get Training Images and masks
        self.train_imgs = self.dataset.train_imgs
        self.train_masks = self.dataset.train_masks
        #get validation images and masks
        self.val_img = self.dataset.val_imgs
        self.val_masks = self.dataset.val_masks
        self.train_img_broken_no = self.dataset.broken_train_imgs_no
        self.val_img_broken_no = self.dataset.broken_val_imgs_no
        
        print("Training and Validation Images loaded...")
        
        #MODEL PARAMETERS
        self.NUM_CLASSES = 6
        self.batch_size = 3
        self.img_size_in = 572
        self.img_size_out = 388
        self.epochs = 7
        self.train_imgs_count = self.imgs_masks_count(self.train_imgs,self.train_masks)
        self.val_imgs_count = self.imgs_masks_count(self.dataset.val_imgs,self.dataset.val_masks)
        self.n_display = 2
        self.save_period = 100
        self.indices = np.array(range(self.train_imgs_count))
        self.val_indices = np.array(range(self.val_imgs_count))
        self.counter_index = 0
        
        #Placeholders
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_in, self.img_size_in, 3], name="X")
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_size_out, self.img_size_out], name="y")
        self.mode = tf.placeholder(tf.bool, name="mode")
        print("PlaceHolders defined..")
        
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op, self.loss_op = None, None
        self.eval_op = None
        self.IOU_op = None
        
    def imgs_masks_count(self,imgs,masks):
        """This functin counts the number of images and masks in the training set,
        and also checks if they are equal in number"""
        
        t_count = len(imgs)
        t_masks_count = len(masks)
        assert t_count == t_masks_count
        return t_count
    
    def make_unet(self, training):
        """Build a U-Net architecture
        Args:
        X (4-D Tensor): (N, H, W, C)
        training (1-D Tensor): Boolean Tensor is required for batchnormalization layers
        Returns:
        output (4-D Tensor): (N, H', W', C')
        Output shape with :
            H' = H - 184
            W' = W - 184
            C' = NUM_CLASSES
        """
        
        #net = X / 127.5 - 1
                
        X = tf.layers.conv2d(self.X, 3, (1, 1), strides = (1,1), padding = 'same',name="color_adjust",reuse = tf.AUTO_REUSE)
                                   
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
        
        return tf.layers.conv2d(conv9, self.NUM_CLASSES, (1, 1), strides = (1,1), name='logits', activation=None, padding='same')
    
    

    def get_next_batch(self,imgs,masks, indices):
        """Return the next images of batch_size in the indices for training"""
        start = self.counter_index
        end = self.counter_index+self.batch_size
        #Also accounting for the case towards the end of the tensor
        img_batch = []
        masks_batch = []
        img_batch.append(imgs[start,:,:,:])
        masks_batch.append(masks[start,:,:])
        
        if self.batch_size>1 and end>self.train_imgs_count:
            excess = end-self.train_imgs_count+1
            temp = random.sample(range(0,start),excess)
        elif (self.batch_size>1):
            temp = indices[start+1:end] #+1 because we are appending 'start[0]' initially and 'end' is NOT included 
        
        if(self.batch_size>1):    
            for i in temp:
                img_batch.append(imgs[i,:,:,:])
                masks_batch.append(masks[i,:,:])
        assert len(img_batch) == self.batch_size == len(masks_batch)
        
        #Cast as np.float32
        img_batch = np.asarray(img_batch)
        img_batch = img_batch.astype(np.float32)
        
        #cast as np.int                
        masks_batch = np.asarray(masks_batch)
        masks_batch = masks_batch.astype(np.int32)
        
        return img_batch,masks_batch
             

    def reset(self):
        """This function resets the counter_index to 0 (for every epoch) 
        and shuffles the indices defined in __init__"""
        
        self.counter_index = 0
        np.random.shuffle(self.indices)
        print("Training Indices Shuffled...")

    def make_train_op(self):
        """Define operations:
            1. Train_op - Adam Optimizer to reduce the cross entropy loss
            2. Loss_op - Op to print the loss
            """
       
        self.pred = self.make_unet(self.mode)
        labels = self.y               
        solver = tf.train.AdamOptimizer(learning_rate = 0.1, epsilon=1e-8)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=labels))
        self.loss_op = loss
        tf.summary.scalar("Loss", tf.Print(self.loss_op, [loss]))
                
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = solver.minimize(loss, global_step=self.global_step)
    
    def IOU(self,preds,labels):
        """
        Inputs:
            preds: Size of (N,H,W,NUM_CLASSES)
            labels: size of (N,H,W,1)
        Output:
            iou_value = a floating point value
        """
        probs = tf.nn.softmax(preds)
        n, h, w, _ = probs.get_shape()
        masks = tf.reshape(probs, [-1, self.NUM_CLASSES])
        masks = tf.reshape(tf.argmax(masks, axis=1), [n.value, h.value, w.value],name = "Prediction")
        iou_value = tf.reduce_sum(tf.cast(tf.equal(tf.cast(masks,tf.int32),labels),dtype = tf.int32))/(n.value*h.value*w.value)
        self.IOU_op = iou_value
        self.IOU_op = tf.Print(self.IOU_op, [self.IOU_op])
        tf.summary.scalar("IOU", self.IOU_op)                    
        
            
    def train(self):
        """Function to Train the Model"""
        
        #Saving directory - WDir/model
        save_dir = os.path.abspath(os.path.join(os.getcwd(),'model','model.ckpt'))
        
        train_logdir = os.path.join(os.getcwd(), "summary","train")
        test_logdir = os.path.join(os.getcwd(),"summary", "test")            
        
        self.make_train_op()
        iou_masks = self.y
        self.IOU(self.pred,iou_masks)
        print("Training Operations defined..")        
        
        summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_summary_writer = tf.summary.FileWriter(test_logdir)
            
            sess.run(tf.global_variables_initializer())
            print("Global Variables defined...")
            saver = tf.train.Saver()
             
            n_display = self.n_display
            global_step_int = tf.train.get_global_step(sess.graph)
            
            for i_epoch in list(range(self.epochs)):
                self.counter_index,sce_loss, n_steps,iou = 0, 0, 0, 0
                self.reset()
                print("Starting epoch {}".format(i_epoch))
                for _ in list(range(0, 5, self.batch_size)):
                    imgs, masks = self.get_next_batch(self.train_imgs,self.train_masks,self.indices)
                    
                    feed_dict = {self.X : imgs,
                                 self.y : masks,
                                 self.mode : True}
                   
                    ops = [self.train_op, global_step_int,self.loss_op,self.IOU_op,summary_op]
                    _, global_step,loss, iou,step_summary = sess.run(ops, feed_dict=feed_dict)
                    
                    train_summary_writer.add_summary(step_summary, global_step)
                    
                    if n_steps + 1 == n_display:
                        print("Cross entropy loss : {}".format(sce_loss/n_steps))
                        print("IOU: {}".format(iou))
                        sce_loss, n_steps,iou = 0, 0,0
                    else:
                        sce_loss += loss
                        n_steps += 1
                        iou+=iou
                    
                    
                    self.counter_index +=self.batch_size
                
                total_iou,self.counter_index = 0, 0    
                for step in range(0, self.val_imgs_count, self.batch_size):
                    X_test, y_test = self.get_next_batch(self.val_img,self.val_masks,self.val_indices)
                    step_iou, step_summary = sess.run([self.IOU_op, summary_op],
                                                      feed_dict={self.X: X_test,
                                                                 self.y: y_test,
                                                                 self.mode: False})
                        
                    total_iou += step_iou * self.batch_size
                    test_summary_writer.add_summary(step_summary, (i_epoch + 1) * (step + 1))
                total_iou /=self.val_imgs_count
                print("IOU for Validation Set = {}".format(total_iou))
                print("Saving model in {} for epoch {}".format(save_dir,i_epoch))
                saver.save(sess, save_dir, global_step)
                    
                print("{} epochs finished.".format(i_epoch))
    
    def join_batch_predictions(self,batch_predictions,broken_no):
        joined_pred = []
        size = self.img_size_out
        for i in broken_no:
            h_no = i[0]
            w_no = i[1]
            a = np.zeros([h_no*size,w_no*size])
            counter = 0 
            for h in range(0,h_no):
                for w in range(0,w_no):
                    a[h:h+size,w:w+size] = batch_predictions[counter,:,:]
                    counter+=1
            joined_pred.append(a)
        return joined_pred
    
    def test(self):
        
        #load the model
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                imported_meta = tf.train.import_meta_graph("model/model-1099.meta")
                imported_meta.restore(sess, tf.train.latest_checkpoint('./model'))
                print(" model restored...")
                X = graph.get_tensor_by_name("X:0")
                print("X restored...")
                predictions = graph.get_tensor_by_name("Prediction:0")
                print("Prediction Restored")
                batch_predictions = []
                actual_pred = []
                X_test_op, y_test_op = tf.train.batch([self.val_img, self.val_masks],
                                                      batch_size=self.batch_size,
                                                      capacity=self.batch_size * 2,
                                                      allow_smaller_final_batch=True)
                
                for step in range(0,self.val_imgs_count,self.batch_size):
                
                    X_batch, Y_batch = sess.run([X_test_op,y_test_op])
                    print("Fetched new batch...")
                    actual_pred.append(Y_batch)
                    feed_dict = {X: X_batch}
                
                    
                    
                    batch_predictions.append(sess.run(predictions, feed_dict = feed_dict))
                    print("Size of batch_predictions is {}".format(len(batch_predictions)))
                    joined_predictions = self.join_batch_predictions(batch_predictions,self.val_img_broken_no)
        return batch_predictions,actual_pred ,np.asarray(joined_predictions)  
    
    def convert_classes_to_color(self,predictions):
        """Colors based on number of classes:
            1 - White - Road
            2 - Blue - House
            3 - Cyan - Low Vegetation
            4 - Green - Vegetation
            5 - Yellow - Car
            6 - Red - Water
            7 - Orange- Unkown
        """
        print("Number of images to convert are {}".format(len(predictions)))
        final = []
        for i in range(len(predictions)):
            a = predictions[i,:,:]
            h,w = a.shape
            b = np.zeros([h,w,3])
            for j in range(0,h):
                for k in range(0,w):
                    #OpenCV works with BGR Format
                    if (a[j,k] == 1):
                        b[j,k,:] = 255 #White if Road
                    elif (a[j,k] == 2):
                        b[j,k,0] = 255 #Blue if house
                    elif (a[j,k] == 3):
                        b[j,k,1] = b[j,k,0] = 255 #Cyan if low Vegetation
                    elif (a[j,k] == 4):
                        b[j,k,1] = 255 #Green if Vegetation
                    elif (a[j,k] == 5):
                        b[j,k,2]=b[j,k,1] = 255 #Yellow if Car
                    elif (a[j,k] == 6):
                        b[j,k,2] =  255 #Red if Water
                    else:
                        b[j,k,0] = 255
                        b[j,k,1] = 165
            final.append(b)
            cv2.imwrite('output_img_{}.png'.format(i),b)
        return final                
        
        
            
model = Model_Unet()
model.train()
#batch_prediction, actual_pred,joined_predictions = model.test()
#final = model.convert_classes_to_color(joined_predictions)