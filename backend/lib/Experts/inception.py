from wandb.keras import WandbCallback
import wandb
from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import os
import sys
import warnings
import numpy as np

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
# set this TensorFlow session as the default session for Keras
tf.compat.v1.keras.backend.set_session(sess)

warnings.filterwarnings('ignore')
sys.path.append(".")


wandb.login()


class NNClassifier:
    def __init__(self, name, input_shape=(128, 128, 3), path_to_file=None):
        self.name = name
        self.input_shape = input_shape
        self.net = self.__create_network(self.input_shape)
        self.lr = 1e-6
        self.net.compile(
            optimizer=Adam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])
        if path_to_file != None:
            self.load_net(path_to_file)
            print('Net loaded')

    # def __custom_sigmoid(self, x):
    #     return 1/(1 + np.e**(-0.3 * x))

    def __create_network(self, input_shape):
        net = InceptionV3(include_top=False,
                          pooling='avg',
                          weights=None,
                          input_tensor=None,
                          input_shape=input_shape)
        x = net.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_layer = Dense(
            1, activation='sigmoid', name='classification')(x)
        net_final = Model(inputs=net.input, outputs=output_layer)
        for layer in net_final.layers:
            layer.trainable = True
        return net_final

    def train(self, generator, eval_data, steps=100, epochs=10):

        wandb.init(project='DP', entity='petttr1',
                   config={
                       "learning_rate": self.lr,
                       "epochs": epochs,
                       "loss_function": "binary_crossentropy",
                       "architecture": "Inception",
                       "dataset": "ICIAR BACH, Binary generator"
                   })

        early_stopping = EarlyStopping(
            patience=30, verbose=1, monitor='val_loss')
        model_checkpoint = ModelCheckpoint(
            './models/{}.hdf5'.format(self.name), save_best_only=True, verbose=1, monitor='val_loss')
        reduce_lr = ReduceLROnPlateau(
            factor=0.01, patience=15, min_lr=1e-9, verbose=1, monitor='val_loss')

        self.net.fit(generator,
                     steps_per_epoch=steps,
                     validation_data=eval_data,
                     epochs=epochs,
                     verbose=1,
                     callbacks=[model_checkpoint, reduce_lr, early_stopping, WandbCallback()])

    def load_net(self, path):
        self.net.load_weights(path)

    def predict(self, image):
        if len(image.shape) < 4:
            image = image[np.newaxis, :]
        return self.net.predict(image)[0][0]
