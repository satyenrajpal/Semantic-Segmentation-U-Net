# Semantic-Segmentation-U-Net

This is a TensorFlow implementation of Semantic Segmentation based on the [U-Net Architecture](https://arxiv.org/abs/1505.04597) for the ISPRS Dataset (vaihingen). 

Requirements-
- Tensorflow
- OpenCV

This code saves the data in a cache file (if the cache file is not present) and then reuses the cached file. Please adhere to the following directory structure:<br />
 
* Bullet list
   * Nested bullet
      * Sub-nested bullet etc
* Bullet list item 2

Argument | Description
--- | --- 
`--env=ENVIRONMENT_NAME`| CartPole-v0, MountainCar-v0 
`--render=1 OR 0` | variable to enable render(1) or not(0)
`--train=1 OR 0` |  variable to train(1) the model or not(0) 
`--type=MODEL_TYPE` | DQN,Dueling
`--save_folder=FOLDER_DIR`| folder directory to save videos (Optional). Videos are not saved if nothing is given
`--model_file=FILE_DIR` | File directory of saved model(Optional). Nothing is done if not given 
To run 


The network expects a 572x572 image input. The architecture is defined as follows-

![](/docs/Architecture.png)




