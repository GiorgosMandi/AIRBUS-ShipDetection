from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class SegmentationModel:

    def __init__(self):
        self.seg_model = None
        self.weights_path = None
        self.callbacks_list = []
        self.name = "unnamed"
        self.model_folder = "../models/"

    def set_callbacks(self):
        print(self.weights_path)
        checkpoint = ModelCheckpoint(self.weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_weights_only=True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=3, verbose=1, mode='min',
                                           epsilon=0.0001, cooldown=0, min_lr=1e-8)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=50)
        self.callbacks_list = [checkpoint, reduceLROnPlat, early]

    def dice_coef(self, y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    # intersection over union
    def IoU(self, y_true, y_pred, eps=1e-6):
        #if tf.math.count_nonzero(y_true) == 0:
        #    return self.IoU(1-y_true, 1-y_pred)
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        return -K.mean( (intersection + eps) / (union + eps + 1), axis=0)

    def compile(self):
        self.seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=self.IoU, metrics=[self.dice_coef, 'binary_accuracy'])

    def load(self):
        self.seg_model.load_weights(self.weights_path)

    def infer(self, x):
        return self.seg_model.predict(x)

    def validate(self, gen, input_len, valid_set, epochs=5, train_steps=200, batch_size=9):
        valid_x = valid_set[0]
        valid_y = valid_set[1]
        step_count = min(train_steps, input_len // batch_size)
        history = [self.seg_model.fit_generator(gen, steps_per_epoch=step_count, epochs=epochs,
                                                validation_data=(valid_x, valid_y),
                                                callbacks=self.callbacks_list,
                                                workers=1  # the generator is not very thread safe
                                                )]
        return history

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
        fig.savefig(self.model_folder + 'results/' + self.name + "_loss.jpg")

    def visualize_validation(self, valid_x, valid_y, load=False):
        if load:
            self.load()
        fig, m_axs = plt.subplots(50, 3, figsize=(20, 200))
        for i, (ax1, ax2, ax3) in enumerate(m_axs):
            test_img = np.expand_dims(valid_x[i], 0)
            y = self.infer(test_img)
            ax1.imshow(valid_x[i])
            ax2.imshow(valid_y[i])
            ax3.imshow(y[0, :, :, 0], vmin=0, vmax=1)
        plt.show()

        fig.savefig(self.model_folder + 'results/' + self.name + "_res.jpg")
