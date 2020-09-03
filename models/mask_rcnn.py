import os
import sys
sys.path.insert(0, os.getcwd())
from utils import get_image, rle_decode, get_colors_for_class_ids
import numpy as np
import matplotlib.pyplot as plt
import random

ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
MASKRCNN_DIR = os.path.join(MODELS_DIR, 'Mask_RCNN')
SERIALIZED_MASKRCNN_DIR = os.path.join(MASKRCNN_DIR, 'serialized')
COCO_WEIGHTS_PATH = os.path.join(MASKRCNN_DIR, 'mask_rcnn_coco.h5.1')
print(MASKRCNN_DIR)
print(COCO_WEIGHTS_PATH)

sys.path.append(MASKRCNN_DIR)

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#import warnings 
#warnings.filterwarnings("ignore")

class MRCNN_Config(Config):    
    NAME = 'Mask-RCNN'
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background and ship classes
    
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 14
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.75 # <-- todo change
    DETECTION_NMS_THRESHOLD = 0.0

    STEPS_PER_EPOCH = 200 
    VALIDATION_STEPS = 50
    
    ## balance out losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 30.0,
        "rpn_bbox_loss": 0.8,
        "mrcnn_class_loss": 6.0,
        "mrcnn_bbox_loss": 1.0,
        "mrcnn_mask_loss": 1.2
    }


class MRCNN:

    def __init__(self, path=None):
        self.weights_path = path
        self.config = MRCNN_Config()
        self.mask_rcnn = modellib.MaskRCNN(mode='training', config=self.config, model_dir=SERIALIZED_MASKRCNN_DIR)
        self.mask_rcnn.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

    def train(self, train, valid, epochs=5, lr=0.0001, layers='all'):
        self.mask_rcnn.train(train, valid, learning_rate=lr,epochs=epochs, layers=layers, augmentation=None)
        history =  self.mask_rcnn.keras_model.history.history
        best_epoch = np.argmin(history["val_loss"])
        score = history["val_loss"][best_epoch]
        print(f'Best Epoch:{best_epoch+1} val_loss:{score}')

        # finding the path of weights of the best epoch of the latest model
        dir_names = next(os.walk(self.mask_rcnn.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)

        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.mask_rcnn.model_dir))

        fps = []
        # Pick last directory
        for d in dir_names: 
            dir_name = os.path.join(self.mask_rcnn.model_dir, d)
            # Find the last checkpoint
            checkpoints = next(os.walk(dir_name))[2]
            checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
            checkpoints = sorted(checkpoints)
            if not checkpoints:
                print('No weight files in {}'.format(dir_name))
            else:
                checkpoint = os.path.join(dir_name, checkpoints[best_epoch])
                fps.append(checkpoint)

        self.weights_path = sorted(fps)[-1]
        print('Found model {}'.format(self.weights_path))
        return history

    # debug this
    def examine_infer(self, dd, n=10):
        infer_config = self.config
        infer_config.IMAGES_PER_GPU = 1
        infer_config.BATCH_SIZE = 1
        # infer_config.display()

        inf_model = modellib.MaskRCNN(mode='inference', config=infer_config, model_dir=MODELS_DIR)
        print("Loading weights from ", self.weights_path)
        inf_model.load_weights(self.weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

        fig = plt.figure(figsize=(10, n*5))
        for i in range(n):
            image_id = random.choice(dd.image_ids)
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dd, infer_config, image_id, use_mini_mask=False)
            plt.subplot(n, 2, 2*i + 1)
            visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dd.class_names, colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
            
            plt.subplot(n, 2, 2*i + 2)
            results = inf_model.detect([original_image]) #, verbose=1)
            r = results[0]
            visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], dd.class_names, r['scores'], 
                                        colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])

    def show_loss(self, loss_history):
        loss_keys = list(loss_history.keys())
        n = len(loss_history.keys())
        fig, axn = plt.subplots(1, n//2, figsize=(10*n//2, 10))
        if n == 2:
            loss = loss_history[loss_keys[0]]
            val_loss = loss_history[loss_keys[1]]
            _ = axn.plot(loss, 'b-', val_loss, 'r-')
            axn.legend(['Training', 'Validation'])
            axn.set_title(loss)
        else:
            for i,ax in enumerate(axn):
                loss = loss_history[loss_keys[i*2]]
                val_loss = loss_history[loss_keys[i*2 + 1]]
                _ = ax.plot(loss, 'b-', val_loss, 'r-')
                ax.legend(['Training', 'Validation'])
                ax.set_title(loss)
        plt.show()



class DetectorDataset(utils.Dataset):
    """Dataset class for training Mask-RCNN.
    """
    def __init__(self, df, shape=(768, 768), img_scaling=(3, 3)):
        super().__init__(self)
        self.img_scaling = img_scaling
        
        # Add classes
        self.add_class('ship', 1, 'Ship')
        # add images 
        for index, row in df.iterrows():
            image_id = row['ImageId']
            count = row['ships']
            mask = row['EncodedPixels']
            self.add_image('ship', image_id=index, path=image_id, annotations=mask, ships=count ,orig_height=shape[0], orig_width=shape[1])
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        image = get_image(path)
        if self.img_scaling is not None:
            image = image[::self.img_scaling[0], ::self.img_scaling[1]]
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = info['ships']

        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.uint8)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.uint8)
            for i, a in enumerate(annotations):
                m = rle_decode(a)
                if self.img_scaling is not None:
                    m = m[::self.img_scaling[0], ::self.img_scaling[1]]
                mask[:, :, i] = m
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.uint8)