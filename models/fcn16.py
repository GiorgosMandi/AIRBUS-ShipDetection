from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2DTranspose
# from keras.layers import Deconvolution2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
import numpy as np
from keras.optimizers import Adam

from models.segmentation_model import SegmentationModel


def Convblock(channel_dimension, block_no, no_of_convs):
    Layers = []
    for i in range(no_of_convs):
        Conv_name = "conv" + str(block_no) + "_" + str(i + 1)

        # A constant kernel size of 3*3 is used for all convolutions
        Layers.append(
            Convolution2D(channel_dimension, kernel_size=(3, 3), padding="same", activation="relu", name=Conv_name))

    Max_pooling_name = "pool" + str(block_no)

    # Addding max pooling layer
    Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=Max_pooling_name))

    return Layers


class FCN16(SegmentationModel):

    def __init__(self, input_shape, model_folder="models/serialized/"):
        super().__init__()
        helper_model = Sequential()
        helper_model.add(Permute((1, 2, 3), input_shape=input_shape))

        for l in Convblock(64, 1, 2): helper_model.add(l)
        for l in Convblock(128, 2, 2): helper_model.add(l)
        for l in Convblock(256, 3, 3): helper_model.add(l)
        for l in Convblock(512, 4, 3): helper_model.add(l)
        for l in Convblock(512, 5, 3): helper_model.add(l)

        helper_model.add(Convolution2D(1024, kernel_size=(7, 7), padding="same", activation="relu", name="fc_6"))

        # Replacing fully connnected layers of VGG Net using convolutions
        helper_model.add(Convolution2D(1024, kernel_size=(1, 1), padding="same", activation="relu", name="fc7"))

        # Gives the classifications scores for each of the 21 classes including background
        helper_model.add(Convolution2D(21, kernel_size=(1, 1), padding="same", activation="relu", name="score_fr"))

        Conv_size = helper_model.layers[-1].output_shape[2]  # 16 if image size if 512
        print("Convolution size: " + str(Conv_size))
        #
        helper_model.add(Conv2DTranspose(21, strides=(2, 2), kernel_size=(4, 4), padding="valid", activation=None, name="score2"))

        # O = ((I-K+2*P)/Stride)+1
        # O = Output dimesnion after convolution
        # I = Input dimnesion
        # K = kernel Size
        # P = Padding

        # I = (O-1)*Stride + K
        Deconv_size = helper_model.layers[-1].output_shape[2]  # 34 if image size is 512*512
        print("Deconvolution size: " + str(Deconv_size))
        # 2 if image size is 512*512

        Extra = (Deconv_size - 2 * Conv_size)
        print("Extra size: " + str(Extra))

        # Cropping to get correct size
        helper_model.add(Cropping2D(cropping=((0, Extra), (0, Extra))))

        Conv_size = helper_model.layers[-1].output_shape[2]

        skip_con = Convolution2D(21, kernel_size=(1, 1), padding="same", activation=None, name="score_pool4")

        # Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
        Summed = add(inputs=[skip_con(helper_model.layers[14].output), helper_model.layers[-1].output])

        Up = Conv2DTranspose(21, kernel_size=(32, 32), strides=(16, 16), padding="valid", activation=None, name="upsample_new")

        # 528 if image size is 512*512
        Deconv_size = (Conv_size - 1) * 16 + 32

        # 16 if image size is 512*512
        extra_margin = (Deconv_size - Conv_size * 16)

        # Cropping to get the original size of the image
        crop = Cropping2D(cropping=((0, extra_margin), (0, extra_margin)))
        self.seg_model = Model(helper_model.input, crop(Up(Summed)))

        self.seg_model.summary()
        self.model_folder = model_folder
        self.weights_path = (model_folder + "fcn_weights.best.hdf5").format('seg_model')
        self.set_callbacks()
        self.name = "FCN16"

    #def compile(self):
    #    self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=self.IoU, metrics=[self.dice_coef, 'binary_accuracy'])
