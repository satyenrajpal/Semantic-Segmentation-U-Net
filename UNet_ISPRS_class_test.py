
import tensorflow as tf
import cv2
#import matplotlib.image as mpimg
import Create_dataset as ct
import numpy as np
from layer_definitions import convert_classes_to_color,get_next_batch,make_unet
import os
import pickle,argparse

class Model_Unet:
    def __init__(self,in_dir,cache_path):
        """
        in_dir: Directory containing the Images and Labelled Masks in the following directory structure 
        (Directory names are within -- --)
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
                            
    def join_batch_predictions(self,batch_predictions,broken_no):
        joined_pred = []
        size = self.img_size_out
        counter = 0 
        for i in broken_no:
            h_no = i[0]
            w_no = i[1]
            a = np.zeros([h_no*size,w_no*size])
            
            for h in range(0,h_no):
                for w in range(0,w_no):
                    a[h*size:(h+1)*size,w*size:(w+1)*size] = batch_predictions[counter]
                    counter+=1
            joined_pred.append(a)
        joined_pred = np.asarray(joined_pred)
        print("Size of joined Predictions = {}".format(joined_pred.shape))
        return joined_pred
    
    def calculate_accuracy(self,batch_pred,actual_label):
        a=np.equal(batch_pred,actual_label)*1
        return np.mean(a)
        
    def get_nxt_test_batch(self,images,masks,counter):
        _end = len(images[0])
        if (counter+self.batch_size <_end):
            req_imgs,req_masks = images[counter:self.batch_size+counter],masks[counter:self.batch_size+counter]
        else:
            req_imgs,req_masks = images[counter:_end],masks[counter:_end]
            diff = self.batch_size-len(req_imgs)
            req_imgs = np.append(req_imgs,images[0:diff],axis=0)
            req_masks= np.append(req_masks,masks[0:diff],axis=0)
        
        assert req_imgs.shape[0] ==self.batch_size
        return req_imgs, req_masks
            
    def test(self,meta_graph,chk_point_dir):
        """
        Uses the Validation images to predict - 
        Outputs:
            actual_predictions - shape: (batch_size, img_size,img_size)
            joined_predictions - shape: (number of images, total_img_size, total_img_size)
        """
        #load the model
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                imported_meta = tf.train.import_meta_graph(meta_graph) ##############
                imported_meta.restore(sess, tf.train.latest_checkpoint(chk_point_dir)) #######################
                print(" model restored...")

                X = graph.get_tensor_by_name("X:0")
                predictions = graph.get_tensor_by_name("Prediction:0")
                mode = graph.get_tensor_by_name("mode:0")
                actual_label = []
                
                counter = 0
                batch_predictions = np.zeros([1,self.img_size_out,self.img_size_out])
                while(counter<self.val_imgs_count):
                    X_batch, Y_batch = self.get_nxt_test_batch(self.val_imgs,self.val_masks,counter)
                    print("Fetched new batch...")
                    
                    actual_label.append(Y_batch)
                    feed_dict = {X: X_batch,
                                 mode: False} 
                    temp = sess.run(predictions,feed_dict=feed_dict)
                    batch_predictions=np.append(batch_predictions,temp,axis=0)
                    print("Size of batch_predictions is {}".format(batch_predictions.shape))
                    counter+=self.batch_size
                
                joined_predictions = self.join_batch_predictions(batch_predictions[1:],self.val_img_broken_no)
                actual_label=np.array(actual_label)
                acc_val = self.calculate_acc(batch_predictions[1:],actual_label)
                print("Accuracy of validation set is {}".format(acc_val))
        return actual_label ,joined_predictions, batch_predictions[1:]
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Network parser- Test')
    parser.add_argument('--cache_file',dest='cache_file',type=str)
    parser.add_argument('--in_dir',dest='in_dir',type=str)
    parser.add_argument('--save_dir'dest='save_dir',type=str,default=os.getcwd())
    parser.add_argument('--meta_file',dest='meta_file',type=str,default=None)
    parser.add_argument('--CP_dir',dest='CP_dir',type=str,default=None)
    
    return parser.parse_args()


def main(args):
    args=parse_arguments()            
    model = Model_Unet(args.in_dir,args.cache_file) #change
    meta_graph=args.meta_file
    chk_point_dir=args.CP_dir
    actual_pred,joined_predictions,batch_pred = model.test(meta_graph,chk_point_dir)
    imgs = convert_classes_to_color(joined_predictions)
    
    # with open('joined_pred_9_2.pkl', mode='wb') as file:
    #     pickle.dump(joined_predictions, file)
    # with open('color_pred_9_2.pkl', mode='wb') as file:
    #     pickle.dump(final, file)
    # with open('batch_pred_9_2.pkl',mode='wb') as file:
    #     pickle.dump(batch_pred,file)
    
    for i,img in enumerate(imgs):
        cv2.imwrite(args.save_dir+'/i.jpg',img)

if __name__ =='__main__':
    main(sys.argv)
