from keras.backend import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
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
        checkpoint = ModelCheckpoint(self.weights_path, monitor='dice_coef', verbose=1, save_best_only=True, mode='max',
                                     save_weights_only=True)
        reduceLROnPlat = ReduceLROnPlateau(monitor='dice_coef', factor=0.33, patience=3, verbose=1, mode='max', epsilon=0.0001, cooldown=0, min_lr=1e-8)
        early = EarlyStopping(monitor="dice_coef", mode="max", patience=50)

        def scheduler(epoch, lr):
            if epoch % 3 != 0:
                return lr
            else:
                return lr * 0.5
        lr_schedule = LearningRateScheduler(schedule=scheduler)
        self.callbacks_list = [checkpoint, reduceLROnPlat, early, lr_schedule]

    def dice_coef(self, y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    # intersection over union
    def IoU(self, y_true, y_pred, eps=1e-6):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
        return K.mean( (intersection + eps) / (union + eps + 1), axis=0)

    def np_IoU(self, y_true, y_pred):
        overlap = y_true * y_pred
        union = y_true + y_pred
        iou = overlap.sum() / float(union.sum())
        return iou

    def np_dice(self, y_true, y_pred):
        return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

    def dice_p_bce(self, in_gt, in_pred):
        return 1e-3 * binary_crossentropy(in_gt, in_pred) - self.dice_coef(in_gt, in_pred)

    def compile(self):
        self.seg_model.compile(optimizer=Adam(1e-2, decay=1e-6), loss=self.dice_p_bce, metrics=[self.dice_coef, self.IoU, 'binary_accuracy'])

    def load(self):
        self.seg_model.load_weights(self.weights_path)

    def infer(self, x):
        return self.seg_model.predict(x)

    def validate(self, gen, input_len, valid_set, epochs=5, train_steps=-1, batch_size=9):
        valid_x = valid_set[0]
        valid_y = valid_set[1]
        if train_steps > 0:
            step_count = train_steps
        else:
            step_count = input_len // batch_size
        print("Steps per epoch: " + str(step_count))
        history = [self.seg_model.fit_generator(gen, steps_per_epoch=step_count, epochs=epochs,
                                                validation_data=(valid_x, valid_y),
                                                callbacks=self.callbacks_list,
                                                workers=1
                                                )]
        return history

    def show_loss(self, loss_history):
        epochs = np.concatenate([mh.epoch for mh in loss_history])
        keys = [k for k in list(loss_history[0].history.keys()) if "val_" not in k]
        fig, axn = plt.subplots(1, len(keys), figsize=(len(keys)*10, 10))
        for i in range(len(keys)):
            if 'val_'+keys[i] in loss_history[0].history.keys():
                _ = axn[i].plot(epochs, np.concatenate([mh.history[keys[i]] for mh in loss_history]), 'b-',
                         epochs, np.concatenate([mh.history['val_'+ keys[i]] for mh in loss_history]), 'r-')
                axn[i].legend(['Training', 'Validation'])
                axn[i].set_title(keys[i])
            else:
                _ = axn[i].plot(epochs, np.concatenate([mh.history[keys[i]] for mh in loss_history]), 'b-')
                axn[i].set_title(keys[i])
        fig.savefig(self.model_folder + 'results/' + self.name + "_loss.jpg")

    def examine_performance(self, valid_x, valid_y, n=40, load=False):
        if load: self.load()
        iou = 0
        dice = 0
        fig, m_axs = plt.subplots(n, 3, figsize=(20, n*10))
        for i, (ax1, ax2, ax3) in enumerate(m_axs):
            test_img = np.expand_dims(valid_x[i], 0)
            y = self.infer(test_img)
            ax1.imshow(valid_x[i])
            ax2.imshow(valid_y[i])
            ax3.imshow(y[0, :, :, 0], vmin=0, vmax=1)
            iou += self.np_IoU(valid_y[i], y[0])
            dice += self.np_dice(valid_y[i], y[0])
        plt.show()
        fig.savefig(self.model_folder + 'results/' + self.name + "_res.jpg")

        print("Average DICE Coefficient: " + str(dice/n))
        print("Average IoU: " + str(iou/n))

