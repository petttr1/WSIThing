from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization, Concatenate, concatenate, add
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import os
import sys
import warnings
import numpy as np
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
# set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)

wandb.login()

warnings.filterwarnings('ignore')
sys.path.append(".")


class NNWeigh:
    def __init__(self, name, num_classes, num_weights=6, input_shape=(256, 256, 3), path_to_file=None):
        self.name = name
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.num_weights = num_weights
        self.lr = 1e-6
        self.models = {}
        self.net = self.__create_network(self.num_weights, self.input_shape)
        self.net.compile(
            optimizer=Adam(lr=self.lr), loss=self.custom_loss, metrics=[self.weighing_acc, self.og_acc, self.acc_diff, self.flip_tpr, self.flip_tnr])
        if path_to_file != None:
            self.load_net(path_to_file)
            print('Net loaded')

    def custom_loss(self, y_true, y_pred):
        true, gt = y_true[:, 0], y_true[:, 1][:, 0]
        weights = K.permute_dimensions(
            K.repeat(y_pred, self.num_classes), (0, 2, 1))
        weighted = true * weights
        summed = K.sum(weighted, axis=1)
        smaxed = K.softmax(summed)
        loss = K.categorical_crossentropy(gt, smaxed)
        return loss

    def weighing_acc(self, y_true, y_pred):
        true, gt = y_true[:, 0], y_true[:, 1][:, 0]
        weights = K.permute_dimensions(
            K.repeat(y_pred, self.num_classes), (0, 2, 1))
        weighted = true * weights
        summed = K.sum(weighted, axis=1)
        smaxed = K.softmax(summed)
        acc = K.mean(K.equal(K.argmax(gt, axis=-1), K.argmax(smaxed, axis=-1)))
        return acc

    def og_acc(self, y_true, y_pred):
        true, gt = y_true[:, 0], y_true[:, 1][:, 0]
        summed = K.sum(true, axis=1)
        smaxed = K.softmax(summed)
        acc = K.mean(K.equal(K.argmax(gt, axis=-1), K.argmax(smaxed, axis=-1)))
        return acc

    def acc_diff(self, y_true, y_pred):
        return self.weighing_acc(y_true, y_pred) - self.og_acc(y_true, y_pred)

    def flip_tpr(self, y_true, y_pred):
        true, gt = y_true[:, 0], y_true[:, 1][:, 0]
        weights = K.permute_dimensions(
            K.repeat(y_pred, self.num_classes), (0, 2, 1))
        weighted = true * weights
        summed = K.sum(weighted, axis=1)
        smaxed = K.softmax(summed)

        og = K.argmax(K.softmax(K.sum(true, axis=1)), axis=-1)
        gts = K.argmax(gt, axis=-1)
        wpred = K.argmax(smaxed, axis=-1)

        ne_gt_wp = K.cast(K.not_equal(gts, wpred), K.floatx())
        ne_gt_og = K.cast(K.not_equal(gts, og), K.floatx())
        e_gt_wp = K.cast(K.equal(gts, wpred), K.floatx())

        # og != gt && weighted == gt
        tp = K.sum(ne_gt_og * e_gt_wp)
        # og != gt && weighted != gt
        fn = K.sum(ne_gt_og * ne_gt_wp)

        return tp / ((tp + fn) + K.epsilon())

    def flip_tnr(self, y_true, y_pred):
        true, gt = y_true[:, 0], y_true[:, 1][:, 0]
        weights = K.permute_dimensions(
            K.repeat(y_pred, self.num_classes), (0, 2, 1))
        weighted = true * weights
        summed = K.sum(weighted, axis=1)
        smaxed = K.softmax(summed)

        og = K.argmax(K.softmax(K.sum(true, axis=1)), axis=-1)
        gts = K.argmax(gt, axis=-1)
        wpred = K.argmax(smaxed, axis=-1)

        ne_gt_wp = K.cast(K.not_equal(gts, wpred), K.floatx())
        e_gt_wp = K.cast(K.equal(gts, wpred), K.floatx())
        e_gt_og = K.cast(K.equal(gts, og), K.floatx())

        #  og == gt && weighted == gt
        tn = K.sum(e_gt_og * e_gt_wp)
        # og == gt && weighted != gt
        fp = K.sum(e_gt_og * ne_gt_wp)

        return tn / ((tn + fp) + K.epsilon())

    def special_sigmoid(self, x):
        return K.sigmoid(x) * 100

    def __create_network(self, num_inputs, input_shape):
        def __make_single_net(shape):
            def residual_module(layer_in, n_filters):
                merge_input = layer_in
                if layer_in.shape[-1] != n_filters:
                    merge_input = Conv2D(n_filters, (1, 1), padding='same',
                                         activation='relu', kernel_initializer='he_normal')(layer_in)
                # conv1
                conv1 = Conv2D(n_filters, (3, 3), padding='same',
                               activation='relu', kernel_initializer='he_normal')(layer_in)
                # conv2
                conv2 = Conv2D(n_filters, (3, 3), padding='same',
                               activation='linear', kernel_initializer='he_normal')(conv1)

                layer_out = add([conv2, merge_input])
                # activation function
                layer_out = Activation('relu')(layer_out)
                return layer_out

            inputs = Input(shape=shape)
            # x = Conv2D(16, 5, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
            #     inputs)  # 128x128x16
            # x = BatchNormalization()(x)
            # x = Conv2D(32, 3, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 64x64x32

            # x = residual_module(x, 32)  # first residual module
            # x = BatchNormalization()(x)

            # x = Conv2D(64, 3, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 32x32x64
            # x = BatchNormalization()(x)
            # x = Conv2D(64, 3, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 16x16x64
            # x = BatchNormalization()(x)
            # x = Conv2D(96, 3, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 8x8x96

            # x = residual_module(x, 96)  # second residual module
            # x = BatchNormalization()(x)

            # x = Conv2D(128, 2, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 4x4x128
            # x = BatchNormalization()(x)
            # x = Conv2D(256, 2, (2, 2), activation='relu',
            #            padding='same', kernel_initializer='he_normal')(x)  # 2x2x256

            # x = residual_module(x, 256)  # third residual module
            # x = BatchNormalization()(x)

            # x = Flatten()(x)
            # x = Dense(512, activation='relu',
            #           kernel_initializer='he_normal')(x)

            x = Conv2D(4, 5, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
                inputs)  # 128x128x16
            x = BatchNormalization()(x)
            x = Conv2D(8, 5, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 64x64x32

            x = residual_module(x, 8)  # first residual module
            x = BatchNormalization()(x)

            x = Conv2D(16, 3, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 32x32x64
            x = BatchNormalization()(x)
            x = Conv2D(16, 3, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 16x16x64
            x = BatchNormalization()(x)
            x = Conv2D(24, 3, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 8x8x96

            x = residual_module(x, 24)  # second residual module
            x = BatchNormalization()(x)

            x = Conv2D(32, 2, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 4x4x128
            x = BatchNormalization()(x)
            x = Conv2D(64, 2, (2, 2), activation='relu',
                       padding='same', kernel_initializer='he_normal')(x)  # 2x2x256

            x = residual_module(x, 64)  # third residual module
            x = BatchNormalization()(x)

            x = Flatten()(x)
            x = Dense(256, activation='relu',
                      kernel_initializer='he_normal')(x)

            out = Dense(1, activation=self.special_sigmoid)(x)
            return Model(inputs=inputs, outputs=out)

        nets = [__make_single_net(input_shape) for _ in range(num_inputs)]
        self.models = {i: n for i, n in zip(range(5, -1, -1), nets)}
        output_layer = concatenate([n.output for n in nets])

        net = Model(inputs=[n.input for n in nets],
                    outputs=output_layer)
        return net

    def train(self, generator, eval_data, steps=100, epochs=10):
        wandb.init(project='DP', entity='petttr1',
                   config={
                       "learning_rate": self.lr,
                       "epochs": epochs,
                       "loss_function": "custom",
                       "architecture": "6-Net",
                       "dataset": "ICIAR BACH"
                   })
        early_stopping = EarlyStopping(
            patience=20, verbose=1, monitor='val_loss')
        model_checkpoint = ModelCheckpoint(
            './models/{}.hdf5'.format(self.name), save_best_only=True, verbose=1, monitor='val_loss')
        reduce_lr = ReduceLROnPlateau(
            factor=0.5, patience=5, min_lr=1e-7, verbose=1, monitor='val_loss')

        config = wandb.config
        config.learning_rate = self.lr

        self.net.fit(generator,
                     validation_data=eval_data,
                     steps_per_epoch=steps,
                     epochs=epochs,
                     verbose=1,
                     callbacks=[model_checkpoint, reduce_lr, early_stopping, WandbCallback()])

    def load_net(self, path):
        self.net.load_weights(path)

    def predict(self, image):
        if len(image.shape) < 4:
            image = image[np.newaxis, :]
        return self.net.predict(image)[0][0]
