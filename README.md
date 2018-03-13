# Semantic-Segmentation-U-Net

This is a TensorFlow implementation of Semantic Segmentation based on the [U-Net Architecture](https://arxiv.org/abs/1505.04597) for the ISPRS Dataset (vaihingen). 

The network expects a 572x572 image input. The architecture is defined as follows-
![](/docs/Architecture.png)

Requirements-
- Tensorflow
- OpenCV

Results - 

Input Image | Output Image
--- | --- 
![](/docs/30.jpg)| ![](/docs/output_0_72.png) 

This code saves the data in a cache file (if the cache file is not present) and then reuses the cached file. Please adhere to the following directory structure:<br />

* in_dir
  * Training Images - (Contains 3 folders)
     * Train - (training Images)
     * Test - Test Images
     * Validation - validation images
  * Labels_classes - (Contains 2 folders)
     * Train - (Training masks)
     * Validation - (Validation Masks)

To run the code run the following command-
`python UNet_ISPRS_class.py` with the following arguments:

Argument | Description
--- | --- 
`--cache_file=CACHE_FILE_PATH`|  If not cached file not present, give any name; a file shall be created   
`--in_dir=DIRECTORY_PATH` | Path to 'in_dir' as described above
`--save_dir=DIR_NAME` |  Directory in which to save model 
`--train_logdir=DIR_NAME` | Directory in which to save training results
`--test_folder=DIR_NAME`| Directory in which to save test results







