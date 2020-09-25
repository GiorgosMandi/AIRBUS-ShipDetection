# AIRBUS-ShipClassification

In this project the goal is to predict the segmentation masks of ships from satellite images. This is a [challenge](https://www.kaggle.com/c/airbus-ship-detection) in 
Kaggle created by AIRBUS. For this purpose, I employ some state of the art semantic segmentation models. 
These are the followings:

* FCN8s
* U-Net
* PSPNet
* Mask-RCNN

The input dataset consists of satellite images capturing the sea, and a CSV Document containing the masks of each image
in the form of RLE. 
In my approach, the images and their masks are loaded through a python generator which is forwarded to an image augmentation generator.
To further pre-process the images, I normalize their values to the range [0,1], downscale them and also re-balancing the
ships distribution in the images by under-sampling the ships-free images and oversampling the images with ships.

Moreover, I use a Learning Rate schedule that halves LR after five epochs. This technique, in addition with reduction of LR on plateau creates
a good training plan that leads to great segmentation masks.
 
To further improve performance, I examine the use of filters that enhances the contrast of the images. In more details I
 use the Histogram Equalization and the Adaptive Equalization. 

## Results 
 |              Model              	|  IoU 	| DICE 	|
|:-------------------------------:	|:----:	|:----:	|
| U-Net                           	| 0.48 	| 0.24 	|
| U-Net + Histogram Equalization  	| 0.42 	| 0.21 	|
| U-Net + Adaptive Equalization   	| 0.44 	| 0.22 	|
| PSPNet                          	| 0.45 	| 0.31 	|
| PSPNet + Histogram Equalization 	| 0.38 	| 0.19 	|
| PSPNet + Adaptive Equalization  	| 0.45 	| 0.23 	|
| Mask-RCNN                       	| 0.45 	| 0.31 	|
  

<img src="https://github.com/GiorgosMandi/AIRBUS-ShipDetection/tree/master/notebooks/results_example.png" alt="Markdown Monster icon"  style="float: left; margin-right: 10px;" />

## How to improve

Due to the lack of available resources, the training of the models was limited. In more details I  have used just a small portion of the initial dataset (just 20K from 200K images)
and train only for 20 epochs. So consider using more images and further train with more epochs. This will probably improve performance as the loss functions do not converge
 after only 20 epochs. So this mean that there is more road to traverse. Furthermore, avoid downscaling the images as higher resolution images will lead to more precise predicted segmentation masks.