# Semantic-Segmentation-U-Net

This is a TensorFlow implementation of Semantic Segmentation based on the [U-Net Architecture](https://arxiv.org/abs/1505.04597) for the ISPRS Dataset (vaihingen). 

Requirements-
- Tensorflow
- OpenCV

This code saves the data in a cache file (if the cache file is not present) and then reuses the cached file. Please adhere to the following directory structure:
`--Images-- (Contains 3 folders)
   -- Train -- (training Images)
   -- Test -- (Test Images)
   -- Validation -- (Validation Images)
 -- Labels_classes -- (Contains 2 folders)
    -- Train -- (Training Masks)
    -- Validation -- (Validation Masks)`


To run 


The network expects a 572x572 image input. The architecture is defined as follows-

![](/docs/Architecture.png)




