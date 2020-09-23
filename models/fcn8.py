from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Cropping2D, Conv2DTranspose,UpSampling2D
from keras.layers import Deconvolution2D
from tensorflow.keras.layers import Input, Add, Dropout, Permute, add
import numpy as np

from models.segmentation_model import SegmentationModel
from tensorflow.keras.optimizers import Adam


def Convblock(channel_dimension, block_no, no_of_convs):
    Layers = []
    for i in range(no_of_convs):
        # A constant kernel size of 3*3 is used for all convolutions
        Conv_name = "conv" + str(block_no) + "_" + str(i + 1)
        Layers.append(Convolution2D(channel_dimension, kernel_size=(3, 3), padding="same", activation="relu", name=Conv_name))
    # Addding max pooling layer
    Max_pooling_name = "pool" + str(block_no)
    Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=Max_pooling_name))

    return Layers


class FCN8(SegmentationModel):

    def __init__(self, input_shape, model_folder="models/serialized/"):
        super().__init__()
        conv_model = Sequential()
        conv_model.add(Permute((1, 2, 3), input_shape=input_shape))

        for l in Convblock(64, 1, 2): conv_model.add(l)
        for l in Convblock(128, 2, 2): conv_model.add(l)
        for l in Convblock(256, 3, 3): conv_model.add(l)
        for l in Convblock(512, 4, 3): conv_model.add(l)
        for l in Convblock(512, 5, 3): conv_model.add(l)

        conv_model.add(Convolution2D(2048, kernel_size=(7, 7), padding="same", activation="relu", name="fc_6"))

        # Replacing fully connected layers of VGG Net using convolutions
        conv_model.add(Convolution2D(2048, kernel_size=(1, 1), padding="same", activation="relu", name="fc7"))

        # Gives the classifications scores for each of the 21 classes including background
        conv_model.add(Convolution2D(1, kernel_size=(1, 1), padding="same", activation="relu", name="score_fr"))

        Conv_size = conv_model.layers[-1].output_shape[2]  # 16 if image size if 512
        print("Convolution size: " + str(Conv_size))
        #
        conv_model.add(Conv2DTranspose(1, strides=(2, 2), kernel_size=(4, 4), padding="valid", activation=None, name="score2"))

        # I = (O-1)*Stride + K
        Deconv_size = conv_model.layers[-1].output_shape[2]  # 34 if image size is 512*512
        print("Deconvolution size: " + str(Deconv_size))
        # 2 if image size is 512*512

        Extra = (Deconv_size - 2 * Conv_size)
        print("Extra size: " + str(Extra))

        # Cropping to get correct size
        conv_model.add(Cropping2D(cropping=((0, Extra), (0, Extra))))

        Conv_size = conv_model.layers[-1].output_shape[2]

        skip_con1 = Convolution2D(1, kernel_size=(1, 1), padding="same", activation=None, name="score_pool4")

        # Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
        Summed = add(inputs=[skip_con1(conv_model.layers[14].output), conv_model.layers[-1].output])

        # Upsampling output of first skip connection
        x = Conv2DTranspose(21, kernel_size=(4, 4), strides=(2, 2), padding="valid", activation=None, name="score4")(Summed)
        x = Cropping2D(cropping=((0, 2), (0, 2)))(x)

        # Conv to be applied to pool3
        skip_con2 = Convolution2D(21, kernel_size=(1, 1), padding="same", activation=None, name="score_pool3")

        # Adding skip connection which takes output og Max pooling layer 3 to current layer
        Summed = add(inputs=[skip_con2(conv_model.layers[10].output), x])

        Up = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), padding="valid", activation=None, name="upsample")(Summed)

        # Cropping the extra part obtained due to transpose convolution
        final = Cropping2D(cropping=((0, 8), (0, 8)))(Up)
        self.seg_model = Model(conv_model.input, final)

        self.seg_model.summary()
        self.model_folder = model_folder
        self.weights_path = (model_folder + "fcn_weights.best.hdf5").format('seg_model')
        self.set_callbacks()
        self.name = "FCN8"

    # def compile(self):
    #     self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=self.dice_p_bce, metrics=[self.dice_coef, self.IoU])