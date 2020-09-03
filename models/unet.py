from keras import models, layers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf

GAUSSIAN_NOISE = 0.1
BATCH_SIZE = 24
MAX_TRAIN_STEPS = 200


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    #if np.max(y_true) == 0.0:
    if tf.math.count_nonzero(y_true) == 0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

class UNet:

    def __init__(self, input_len):
        input_img = layers.Input(input_len, name='RGB_Input')
        pp_in_layer = input_img
        pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
        pp_in_layer = layers.BatchNormalization()(pp_in_layer)

        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(pp_in_layer)
        c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        u8 = layers.UpSampling2D((2, 2))(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

        u9 = layers.UpSampling2D((2, 2))(c8)
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

        d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.seg_model = models.Model(inputs=[input_img], outputs=[d])
        self.seg_model.summary()

    def compile(self):
        self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])
                               # loss=dice_p_bce, metrics=[dice_coef, 'binary_accuracy'])

    def validate(self, gen, input_len, valid_set, epochs=5, train_steps=MAX_TRAIN_STEPS):
        weight_path = "models/serialized/unet_weights.best.hdf5".format('seg_model')

        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min', save_weights_only=True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33,patience=3,verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=1e-8)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        callbacks_list = [checkpoint, early, reduceLROnPlat]

        valid_x = valid_set[0]
        valid_y = valid_set[1]

        step_count = min(train_steps, input_len // BATCH_SIZE)
        history = [self.seg_model.fit_generator(gen, steps_per_epoch=step_count, epochs=epochs,
                                                     validation_data=(valid_x, valid_y),
                                                     callbacks=callbacks_list,
                                                     workers=1  # the generator is not very thread safe
                                                     )]
        return history

    def load(self):
        self.seg_model = self.seg_model.load_weights(weight_path)

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

