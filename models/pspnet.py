from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Convolution2D, BatchNormalization, LeakyReLU, ReLU, GlobalAveragePooling2D, UpSampling2D, \
    AveragePooling2D, Activation, Add
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


GAUSSIAN_NOISE = 0.1
BATCH_SIZE = 1
MAX_TRAIN_STEPS = 200

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if tf.math.count_nonzero(y_true) == 0:
        return IoU(1-y_true, 1-y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

class PSPNet:
    def __init__(self, input_shape):
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
            base = conv_block(input_layer, [32, 32, 64], '1')
            # block_2
            base = conv_block(base, [64, 64, 128], '2')
            # block_3
            base = conv_block(base, [128, 128, 256], '3')
            return base

        def pyramid_feature_maps(input_layer):
            # pyramid pooling module

            base = base_feature_maps(input_layer)
            # red
            red = GlobalAveragePooling2D(name='red_pool')(base)
            red = tf.keras.layers.Reshape((1, 1, 256))(red)
            red = Convolution2D(filters=64, kernel_size=(1, 1), name='red_1_by_1')(red)
            red = UpSampling2D(size=256, interpolation='bilinear', name='red_upsampling')(red)
            # yellow
            yellow = AveragePooling2D(pool_size=(2, 2), name='yellow_pool')(base)
            yellow = Convolution2D(filters=64, kernel_size=(1, 1), name='yellow_1_by_1')(yellow)
            yellow = UpSampling2D(size=2, interpolation='bilinear', name='yellow_upsampling')(yellow)
            # blue
            blue = AveragePooling2D(pool_size=(4, 4), name='blue_pool')(base)
            blue = Convolution2D(filters=64, kernel_size=(1, 1), name='blue_1_by_1')(blue)
            blue = UpSampling2D(size=4, interpolation='bilinear', name='blue_upsampling')(blue)
            # green
            green = AveragePooling2D(pool_size=(8, 8), name='green_pool')(base)
            green = Convolution2D(filters=64, kernel_size=(1, 1), name='green_1_by_1')(green)
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
        self.weight_path = "models/serialized/pspnet_weights.best.hdf5".format('seg_model')

    def compile(self):
        self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=IoU, metrics=[dice_coef, 'binary_accuracy'])

    def validate(self, gen, input_len, valid_set, epochs=5, train_steps=MAX_TRAIN_STEPS):
        checkpoint = ModelCheckpoint(self.weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=3, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=1e-8)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        callbacks_list = [checkpoint, early, reduceLROnPlat]

        valid_x = valid_set[0]
        valid_y = valid_set[1]
        steps = input_len // BATCH_SIZE
        step_count = min(train_steps, steps)
        history = [self.seg_model.fit_generator(gen, steps_per_epoch=step_count, validation_steps=step_count, epochs=epochs,
                                                validation_data=(valid_x, valid_y),
                                                callbacks=callbacks_list,
                                                workers=1  # the generator is not very thread safe
                                                )]
        return history

    def load(self):
        self.seg_model = self.seg_model.load_weights(self.weight_path)

    def infer(self, x):
        return self.seg_model.predict(x)

    def show_loss(self, loss_history):
        epochs = np.concatenate([mh.epoch for mh in loss_history])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

        _ = ax1.plot(epochs, np.concatenate([mh.history['loss'] for mh in loss_history]), 'b-',
                     epochs, np.concatenate([mh.history['val_loss'] for mh in loss_history]), 'r-')
        ax1.legend(['Training', 'Validation'])
        ax1.set_title('Loss')

        _ = ax2.plot(epochs, np.concatenate([mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                     epochs, np.concatenate([mh.history['val_binary_accuracy'] for mh in loss_history]), 'r-')
        ax2.legend(['Training', 'Validation'])
        ax2.set_title('Binary Accuracy (%)')