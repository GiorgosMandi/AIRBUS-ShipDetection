
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Convolution2D, BatchNormalization, LeakyReLU, ReLU, GlobalAveragePooling2D, UpSampling2D, \
    AveragePooling2D, Activation, Add
import tensorflow as tf
from keras.layers import MaxPooling2D, ZeroPadding2D, Concatenate

from models.segmentation_model import SegmentationModel

GAUSSIAN_NOISE = 0.1
BATCH_SIZE = 1
MAX_TRAIN_STEPS = 500



class PSPNet(SegmentationModel):
    def __init__(self, input_shape, model_folder="models/serialized/"):
        super().__init__()

        def conv_block(X, filters, block):
            # resiudal block with dilated convolutions
            # add skip connection at last after doing convoluion operation to input X

            b = 'block_' + str(block) + '_'
            f1, f2, f3 = filters
            X_skip = X
            # block_a
            X = Convolution2D(filters=f1, kernel_size=(1, 1), dilation_rate=(1, 1),
                              padding='same', kernel_initializer='he_normal', name=b + 'a')(X)
            X = BatchNormalization(name=b + 'batch_norm_a')(X)
            X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_a')(X)
            # block_b
            X = Convolution2D(filters=f2, kernel_size=(3, 3), dilation_rate=(2, 2),
                              padding='same', kernel_initializer='he_normal', name=b + 'b')(X)
            X = BatchNormalization(name=b + 'batch_norm_b')(X)
            X = LeakyReLU(alpha=0.2, name=b + 'leakyrelu_b')(X)
            # block_c
            X = Convolution2D(filters=f3, kernel_size=(1, 1), dilation_rate=(1, 1),
                              padding='same', kernel_initializer='he_normal', name=b + 'c')(X)
            X = BatchNormalization(name=b + 'batch_norm_c')(X)
            # skip_conv
            X_skip = Convolution2D(filters=f3, kernel_size=(3, 3), padding='same', name=b + 'skip_conv')(X_skip)
            X_skip = BatchNormalization(name=b + 'batch_norm_skip_conv')(X_skip)
            # block_c + skip_conv
            X = Add(name=b + 'add')([X, X_skip])
            X = ReLU(name=b + 'relu')(X)
            return X

        def base_feature_maps(input_layer):
            # base convolution module to get input image feature maps

            # block_1
            #base = conv_block(input_layer, [32, 32, 64], '1')
            base = conv_block(input_layer, [8, 8, 8], '1')

            # block_2
            #base = conv_block(base, [64, 64, 128], '2')
            base = conv_block(base, [16, 16, 16], '2')

            # block_3
            #base = conv_block(base, [128, 128, 256], '3')
            base = conv_block(base, [32, 32, 32], '3')

            return base

        def pyramid_feature_maps(input_layer):
            # pyramid pooling module
            # [8,128,192,192]
            base = base_feature_maps(input_layer)
            # red
            red = GlobalAveragePooling2D(name='red_pool')(base)
            red = tf.keras.layers.Reshape((1, 1, 32))(red)
            red = Convolution2D(filters=16, kernel_size=(1, 1), name='red_1_by_1')(red)
            red = UpSampling2D(size=256, interpolation='bilinear', name='red_upsampling')(red)
            # yellow
            yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
            yellow = Convolution2D(filters=16, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
            yellow = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow)
            # blue
            blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base)
            blue = Convolution2D(filters=16, kernel_size=(1, 1), name='blue_1_by_1')(blue)
            blue = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue)
            # green
            green = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base)
            green = Convolution2D(filters=16, kernel_size=(1, 1), name='green_1_by_1')(green)
            green = UpSampling2D(size=8, interpolation='bilinear', name='green_upsampling')(green)
            # base + red + yellow + blue + green
            return tf.keras.layers.concatenate([base, red, yellow, blue, green])

        def last_conv_module(input_layer):
            X = pyramid_feature_maps(input_layer)
            X = Convolution2D(filters=3, kernel_size=3, padding='same', name='last_conv_3_by_3')(X)
            X = BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
            X = Activation('sigmoid', name='last_conv_relu')(X)
            X = Convolution2D(1, (1, 1), activation='sigmoid')(X)
            return X

        input_layer = tf.keras.Input(shape=input_shape, name='input')
        output_layer = last_conv_module(input_layer)
        self.seg_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        self.seg_model.summary()
        self.model_folder = model_folder
        self.weights_path = (model_folder + "pspnet_weights.best.hdf5").format('seg_model')
        self.set_callbacks()
        self.name = "PSPNet"



    def compile(self):
        self.seg_model.compile(optimizer=SGD(1e-4, decay=1e-6, momentum=0.9, nesterov=True), loss=self.dice_p_bce, metrics=[self.dice_coef, self.IoU, 'binary_accuracy'])
