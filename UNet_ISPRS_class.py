
import tensorflow as tf
import Create_dataset as ct
import numpy as np
from layer_definitions import get_next_batch,make_unet
import os
import argparse

class Model_Unet:
    def __init__(self,in_dir,cache_path):
        """
        in_dir: Directory containing the Images and Labelled Masks in the following directory structure 
        (Directory names are between --   --)
        --Images-- (Contains 3 folders)
            -- Train -- (training Images)
            -- Test -- (Test Images)
            -- Validation -- (Validation Images)
        -- Labels_classes -- (Contains 2 folders)
            -- Train -- (Training Masks)
            -- Validation -- (Validation Masks)
        """
        self.dataset = ct.cache(cache_path=cache_path, fn=ct.Dataset,in_dir=in_dir)
        #get Training Images and masks
        self.train_imgs = self.dataset.train_imgs
        self.train_masks = self.dataset.train_masks
        #get validation images and masks
        self.val_imgs = self.dataset.val_imgs
        self.val_masks = self.dataset.val_masks
        self.train_img_broken_no = self.dataset.broken_train_imgs_no
        self.val_img_broken_no = self.dataset.broken_val_imgs_no
       
        print("Training and Validation Images loaded...")
        
        #MODEL PARAMETERS
        self.NUM_CLASSES = 6
        self.batch_size = 4
        self.img_size_in = 572
        self.img_size_out = 388
        self.epochs = 80
        self.train_imgs_count = self.imgs_masks_count(self.train_imgs,self.train_masks)
        self.val_imgs_count = self.imgs_masks_count(self.dataset.val_imgs,self.dataset.val_masks)
        self.n_display = 10
        self.save_period = 100
        
        #Placeholders
        self.X = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size_in, self.img_size_in, 3], name="X")
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.img_size_out, self.img_size_out], name="y")
        #We wish to train the model with sparse labels but then validate it with fully labelled dataset
        
        self.mode = tf.placeholder(tf.bool, name="mode")
        print("PlaceHolders defined..")
        
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op, self.loss_op = None, None
        self.eval_op = None
        self.IOU_op,self.masks=None,None
        
        
    def imgs_masks_count(self,imgs,masks):
        """This functin counts the number of images and masks in the training set,
        and also checks if they are equal in number"""
        
        t_count = len(imgs)
        t_masks_count = len(masks)
        assert t_count == t_masks_count
        return t_count
    
    def make_train_op(self):
        """Define operations:
            1. Train_op - Adam Optimizer to reduce the cross entropy loss
            2. Loss_op - Op to print the loss
            """
        labels = self.y    
        solver = tf.train.AdamOptimizer()           
        self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pred, labels=labels))
        tf.summary.scalar("Loss", self.loss_op)
                
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = solver.minimize(self.loss_op, global_step=self.global_step)
        
    def IOU(self,preds,labels):
        """
        Inputs:
            preds: Size of (N,H,W,NUM_CLASSES)
            labels: size of (N,H,W)
        Output:
            iou_value = a floating point value
        """
        probs = tf.nn.softmax(preds)
        n, h, w, _ = probs.get_shape()
        masks = tf.reshape(probs, [-1, self.NUM_CLASSES])
        self.masks = tf.reshape(tf.argmax(masks, axis=1), [n.value, h.value, w.value],name = "Prediction")
        self.masks = tf.cast(self.masks,tf.int32)
        equal_as_int = tf.cast(tf.equal(self.masks,labels),tf.float32)
        self.IOU_op = tf.reduce_mean(equal_as_int)
        tf.summary.scalar("IOU", self.IOU_op)                    
        
    def evaluate_acc_class(self,batch_pred,classes,labels):
        """Evaluate class-wise accuracy"""
        
        acc_classes = np.zeros(classes)
        for i in range(classes):
            idx_i = np.where(labels==i) 
            batch_i = batch_pred[idx_i] #batch_size is a 1-D array
            if batch_i.size!=0:
                acc_classes[i] = np.where(batch_i==i)[0].size/idx_i[0].size
                
        return acc_classes
    
    def train(self,save_dir,train_logdir,test_logdir,train_random):
        """Function to Train the Model"""
        
        val_imgs_T = tf.convert_to_tensor(self.val_imgs)
        val_masks_T = tf.convert_to_tensor(self.val_masks)
        X_test_op, y_test_op = tf.train.batch([val_imgs_T, val_masks_T],
                                          batch_size=self.batch_size,
                                          capacity=self.batch_size * 2,
                                          enqueue_many = True)
        
        self.pred = make_unet(self.X, self.mode,self.NUM_CLASSES)
        self.IOU(self.pred,self.y)
        self.make_train_op()
        print("Training Operations defined..")
        summary_op = tf.summary.merge_all()
        
        with tf.Session() as sess:
            train_summary_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_summary_writer = tf.summary.FileWriter(test_logdir,sess.graph)
            train_random_summary_writer = tf.summary.FileWriter(train_random,sess.graph)
            
            sess.run(tf.global_variables_initializer())
            print("Global Variables defined...")
            saver = tf.train.Saver()
            max_IOU, n_display = 0, self.n_display
       
            try:
                global_step = tf.train.get_global_step(sess.graph)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
        
                for i_epoch in list(range(self.epochs)):
                    sce_loss, n_steps,iou_batch = 0, 0, 0
                    print("Starting epoch {}".format(i_epoch+1))
                    
                    """Training Set"""
                    for _ in list(range(0, self.train_imgs_count, self.batch_size)):
                        imgs, masks,idx_len = get_next_batch(self.train_imgs,self.train_masks,self.batch_size)

                        ops = [self.train_op, global_step,self.loss_op,self.IOU_op,summary_op]
                        __, global_step_int,loss, iou,step_summary = sess.run(ops, feed_dict={self.X : imgs,
                                                                                             self.y : masks,
                                                                                             self.mode : True})
                        train_summary_writer.add_summary(step_summary, global_step_int)
                                                
                        if n_steps + 1 == n_display:
                            #Obtain prediction of batch
                            batch_pred = sess.run(self.masks,feed_dict={self.X:imgs,
                                                                       self.y : masks,
                                                                       self.mode : False})
                            with  open('Classes_acc.txt','a') as acc_file:
                                acc_batch_cl=self.evaluate_acc_class(batch_pred,self.NUM_CLASSES,masks) #Calculate acc classwise for batch
                                print(acc_batch_cl)
                                acc_file.write('{} - {} \n'.format(global_step_int,acc_batch_cl))
                            print("Cross entropy loss : {}".format(sce_loss/n_steps))
                            print("IOU: {}".format(iou_batch/n_steps))
                            sce_loss, n_steps,iou_batch = 0, 0, 0
                        else:
                            sce_loss += loss
                            n_steps += 1
                            iou_batch+=iou
                        
                    """Validation Set"""
                    n_steps,total_iou = 0,0
                    class_acc_val=np.zeros(self.NUM_CLASSES)
                    for step in range(0, self.val_imgs_count, self.batch_size):
                        
                        X_test, y_test = sess.run([X_test_op, y_test_op])
                        step_iou, step_summary,val_pred = sess.run([self.IOU_op, summary_op,self.masks],
                                                          feed_dict={self.X: X_test,
                                                                     self.y: y_test,
                                                                     self.mode: False})
                        class_acc_val+=self.evaluate_acc_class(val_pred,self.NUM_CLASSES,y_test)                    
                        total_iou += step_iou
                        n_steps+=1
                        _step =i_epoch*np.ceil(self.val_imgs_count/self.batch_size) + n_steps
                        test_summary_writer.add_summary(step_summary, int(_step))
                    total_iou /= n_steps
                    class_acc_val/=n_steps
                    print("IOU for Validation Set = {}".format(total_iou))
                    print("class-wise accuracy-{}".format(class_acc_val))
                    with open('test_IOU.txt', 'a') as the_file:
                        the_file.write('Val_IOU for {} epoch is- {} \n'.format(i_epoch+1,total_iou))
                        if(class_acc_val[4]>0.6):
                                the_file.write('^car acc - {} for {}\n'.format(class_acc_val[4],global_step_int))
                                saver.save(sess, save_dir,global_step_int) 
                    
                    if(total_iou>max_IOU):
                        saver.save(sess, save_dir,global_step_int) #Save only if total_iou improves
                        print("Saving model in {} for epoch {}".format(save_dir,i_epoch+1))
                        max_IOU = total_iou
                        
                    """ Random val_count images from training Set"""
                    n_steps,total_iou = 0,0           
                    for step in range(0,self.val_imgs_count,self.batch_size):
                        X_train_inf, Y_train_inf,idx_len = get_next_batch(self.train_imgs,self.train_masks,self.batch_size)
                        
                        step_iou, step_summary = sess.run([self.IOU_op, summary_op],
                                                          feed_dict={self.X: X_train_inf,
                                                                     self.y: Y_train_inf,
                                                                     self.mode: False})
                        total_iou += step_iou
                        n_steps+=1
                        _step =i_epoch*np.ceil(self.val_imgs_count/self.batch_size) + n_steps
                        train_random_summary_writer.add_summary(step_summary, _step)
                    total_iou /= n_steps
                    print("IOU for {} random training Set images = {}".format(self.val_imgs_count,total_iou))
                    with open('train_random.txt', 'a') as the_file:
                        the_file.write('IOU for {} epoch is- {} \n'.format(i_epoch+1,total_iou))
                    
                    print("{} epochs finished.".format(i_epoch+1))
            
            finally:
                coord.request_stop()
                coord.join(threads)
                saver.save(sess, save_dir)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Semantic Segmentation Network parser')
	parser.add_argument('--cache_file',dest='cache_file',type=str)
	parser.add_argument('--in_dir',dest='in_dir',type=str)
	parser.add_argument('--save_dir'dest='save_dir',type=str,default=os.getcwd())
	parser.add_argument('--train_logdir',dest='train_logdir',default=os.path.join('train',os.getcwd()))
	parser.add_argument('--test_logdir',dest='test_logdir',type=str,default=os.path.join('test',os.getcwd()))
	#parser.add_argument('--model_file',dest='model_file',type=str,default=None)
	return parser.parse_args()

    
def main(args):            
	args=parse_arguments()
    model = Model_Unet(args.in_dir,args.cache_file) #change
    #save_dir = '/data/satyenr/model_16_2/model.ckpt'
    #train_logdir = '/data/satyenr/summary_16_2/train' 
    #test_logdir = '/data/satyenr/summary_16_2/test'            
    #train_random = '/data/satyenr/summary_16_2/train_rand' 
    train_random=args.train_logdir+'_rand'
    model.train(args.save_dir,args.train_logdir,args.test_logdir,train_random)

if __name__ =='__main__':
    main(sys.argv)