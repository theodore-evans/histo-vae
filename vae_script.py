#!/usr/bin/env python3
# adapted from https://keras.io/examples/generative/vae/
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd
import random
import h5py

import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    '--logs-dir', type=Path, default=None,
    help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)
parser.add_argument(
    '--weights-dir', type=Path, default=None,
    help='Debug: Path to save the weights'
)
parser.add_argument(
    '-dim', type=int, default=1024,
    help='The size of latent dimension representation'
)
parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch Size'
)

args = parser.parse_args()
weights_dir = args.weights_dir.expanduser()
weights_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = Path(weights_dir, 'weights.ckpt')

if args.logs_dir:
    log_file_path = Path(args.logs_dir, 'train.log').expanduser()
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

print('List GPUs:', tf.config.list_physical_devices('GPU'))

# %%


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def reconstruct(self, data):
        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        return reconstruction

    def call(self, data, training=False):
        if isinstance(data, tuple):
            data = data[0]
        reconstruction = decoder(z)
        return reconstruction

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(data)
            reconstruction = decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 32 * 32
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


# %%
cp_callback = callbacks.ModelCheckpoint(filepath=str(checkpoint_file),
                                        save_weights_only=True)
es_callback = callbacks.EarlyStopping(
    monitor="loss",
    patience=5,
)
# %%
latent_dim = args.dim

encoder_inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(64, kernel_size=4, activation="relu",
                  strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, kernel_size=4, activation="relu",
                  strides=2, padding="same")(x)
x = layers.Conv2D(512, kernel_size=4, activation="relu",
                  strides=2, padding="same")(x)
encoded = layers.Flatten()(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(encoded)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoded)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(2048)(latent_inputs)
x = layers.Reshape((4, 4, 128))(x)
x = layers.Conv2DTranspose(
    256, kernel_size=4, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(
    64, kernel_size=4, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(
    3, kernel_size=4, activation="relu", strides=2, padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# %%
TRAIN_X_FILE = 'camelyonpatch_level_2_split_train_x.h5'
train_x = h5py.File(Path('./data', TRAIN_X_FILE), 'r',
                    libver='latest', swmr=True)['x']


def data_generator(x, batch_size=None):
    index = range(len(x))

    while True:
        index_sample = index
        if batch_size is not None:
            index_sample = sorted(random.sample(index, batch_size))
        x_pos = random.randint(0, 64)
        y_pos = random.randint(0, 64)
        x_data = x[index_sample, x_pos:x_pos+32, y_pos:y_pos+32] / 255.0
        yield x_data

data_size = len(train_x)
batch_size = args.batch_size
steps_per_epoch = data_size // batch_size

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(data_generator(train_x, batch_size=batch_size),
        epochs=100,
        steps_per_epoch = steps_per_epoch,
        batch_size=batch_size,
        callbacks=[cp_callback, es_callback])

# %%

if args.logs_dir:
    sys.stdout.close()
