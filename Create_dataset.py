import numpy as np
import os
import pickle
import cv2
#import matplotlib.image as mpimg

def cache(cache_path, fn, *args, **kwargs):
    """
    Cache-wrapper for a function or class. If the cache-file exists
    then the data is reloaded and returned, otherwise the function
    is called and the result is saved to cache. The fn-argument can
    also be a class instead, in which case an object-instance is
    created and saved to the cache-file.
    :param cache_path:
        File-path for the cache-file.
    :param fn:
        Function or class to be called.
    :param args:
        Arguments to the function or class-init.
    :param kwargs:
        Keyword arguments to the function or class-init.
    :return:
        The result of calling the function or creating the object-instance.
    """
        
    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


class Dataset:
    def __init__(self,in_dir):

        self.exts_imgs = '.jpg'
        self.exts_ann = '.png'
        
        self.exts_imgs = tuple(ext.lower() for ext in self.exts_imgs)
        self.exts_ann = tuple(ext.lower() for ext in self.exts_ann)
        self.in_dir  = in_dir;
        in_dir_full = in_dir;
        img_size = 572
        ann_size = 388
        
        train_img_path = os.path.join(in_dir_full,'Images','Train')
        train_ann_path = os.path.join(in_dir_full,'Labels_classes','Train')
        val_img_path = os.path.join(in_dir_full,'Images','Validation')
        val_ann_path = os.path.join(in_dir_full,'Labels_classes','Validation')
        
        if os.path.isdir(train_img_path):
            self.filenames_img = self._get_filenames(train_img_path,self.exts_imgs)
        else:
            assert 0==1
                        
        if os.path.isdir(train_ann_path):
            self.filenames_ann = self._get_filenames(train_ann_path,self.exts_ann)
            print("Training Images file names Loaded!")
            
        if os.path.isdir(val_img_path):
            self.filenames_val_imgs = self._get_filenames(val_img_path,self.exts_imgs)
            
        if os.path.isdir(val_ann_path):
            self.filenames_val_ann = self._get_filenames(val_ann_path,self.exts_ann)
            print("Validation Images file names Loaded!")
            
        self.train_imgs,self.broken_train_imgs_no = self.load_images(self.filenames_img,ann_size,img_size,RGB_padding = True)
        self.train_masks,_ = self.load_images(self.filenames_ann,ann_size,img_size,RGB_padding = False)
        print("Training images loaded with shape: {}".format(self.train_imgs.shape))
        self.val_imgs, self.broken_val_imgs_no= self.load_images(self.filenames_val_imgs,ann_size,img_size,RGB_padding = True)
        print("Validation images loaded with shape: {}".format(self.val_imgs.shape))
        self.val_masks,_ = self.load_images(self.filenames_val_ann,ann_size,img_size,RGB_padding = False)
        
        #Remove 7 from validation and training images
        self.train_masks = self.remove_label(7,6,self.train_masks) 
        self.val_masks = self.remove_label(7,6,self.val_masks)
        
        print("Training masks loaded with shape: {} and as type {}".format(self.train_masks.shape,self.train_masks.dtype))
        print("Validation masks loaded with shape: {} and as type {}".format(self.val_masks.shape,self.train_masks.dtype))
        
    def _get_filenames(self, dir,exts):
        """
        Create and return a list of filenames with matching extensions in the given directory.
        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.
        :return:
            List of filenames. Only filenames. Does not include the directory.
        """
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(exts):
                    filenames.append(os.path.join(os.path.abspath(dir),filename))
        return filenames
        
    def load_images(self,image_paths,size,size_wpadding,RGB_padding = True):
    # Load the images from disk.
        images_broken = []
        broken_no = []
        one_side_pad = int((size_wpadding-size)/2)
        
        for path in image_paths:
            if RGB_padding:
                image = cv2.imread(path,1)
                shp = image.shape
                w_no = (shp[1]-2*one_side_pad)//size
                h_no = (shp[0]-2*one_side_pad)//size
                
                for i in range(0,int(h_no)):
                  for j in range(0,int(w_no)):
                      broken_img = image[i*size:i*size+size_wpadding,
                                         j*size:j*size+size_wpadding,:]
                      images_broken.append(broken_img)
            else:
                image = cv2.imread(path,0)
                shp = image.shape
                cropped_img = image[one_side_pad:, one_side_pad:]
                cropped_img = cropped_img[:-one_side_pad,:-one_side_pad]
                shp = cropped_img.shape
                w_no = (shp[1])//size
                h_no = (shp[0])//size
                
                for i in range(0,int(h_no)):
                    for j in range(0,int(w_no)):
                        broken_img = cropped_img[i*size:(i+1)*size,
                                                 j*size:(j+1)*size]
                        images_broken.append(broken_img)
            broken_no.append(np.array([h_no,w_no],dtype=np.int32))
        # Convert to a numpy array and returns it in the form of [num_images,size,size,channel]
        return np.asarray(images_broken), broken_no

    def remove_label(self,in_label,out_label,masks):
        """ Masks: Shape - (batch_size,img_size,img_size)
        """
        masks[np.where(masks==in_label)]=out_label
        assert len(np.where(masks == in_label)[0]) == 0
        masks = masks-np.ones(masks.shape)
        
        return masks