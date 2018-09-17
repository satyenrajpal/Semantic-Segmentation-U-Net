# Frugal-Semantic-Segmentation-U-Net

This is a TensorFlow implementation of Semantic Segmentation based on the [U-Net Architecture](https://arxiv.org/abs/1505.04597) for the ISPRS Dataset (vaihingen) with an improvement to help predict class segments that constitute a small proportional of all labels. Essentially, the idea is to remove labels of classes that constitute a large proportion of the dataset, in other words reduce bias. <br>
In this project, we optimized over 35.29% (programatically reducing labels corresponding to 'house', 'vegetation' and 'road' classes) of the complete labels. As seen in the results, when optimizing over all labels, the network is unable to segment 'cars' (yellow) pixels. However, when optimizing over a pruned dataset not only are the pixels for cars predicted, we achieve an accuracy of 78%.

The network expects a 572x572 image input. The architecture is defined as follows-
![](/docs/Architecture.png)

Requirements-
- Tensorflow
- OpenCV

This code saves the data in a cache file (if the cache file is not present) and then reuses the cached file. If the cache file is present, it creates one and treats it as the dataset. <br>

Please adhere to the following directory structure:<br />

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


Results - 
Conventional Semantic Segmentation <br>
Input Image | Conventional Segmentation 
 --- | --- 
![](/docs/30.jpg)| ![](/docs/ouptut_0_72.png) 







