#!/usr/bin/env python3
# adapted from https://keras.io/examples/generative/vae/
# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras, Tensor
from tensorflow.keras import layers, callbacks
from tensorflow.keras.constraints import UnitNorm
import pandas as pd
import random
import h5py

import sys
import argparse
from pathlib import Path

#%%
parser = argparse.ArgumentParser()
parser.add_argument(
    '--logs-dir', type=Path, default=None,
    help='Debug: Path to writable directory for a log file to be created. Default: log to stdout / stderr'
)
parser.add_argument(
    '--weights-vae', type=str, default=None,
    help='Debug: Path to load the weights of vae'
)
parser.add_argument(
    '--weights-dir', type=Path, default='./weights',
    help='Debug: Path to save the weights'
)
parser.add_argument(
    '-dim', type=int, default=128,
    help='The size of latent dimension representation'
)
parser.add_argument(
    '--batch-size', type=int, default=128,
    help='Batch Size'
)

args = parser.parse_args()
weights_dir = args.weights_dir.expanduser()
weights_dir.mkdir(parents=True, exist_ok=True)
checkpoint_file = Path(weights_dir, 'checkpoint.ckpt')

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
k_dim = latent_dim

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

#%%
def relu_bn(inputs: Tensor) -> Tensor:
    relu = layers.ReLU()(inputs)
    bn = layers.BatchNormalization()(relu)
    return bn

def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = layers.Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same")(x)
    y = relu_bn(y)
    y = layers.Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same")(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same")(x)
    out = layers.Add()([x, y])
    out = relu_bn(out)
    return out

def create_res_net():
    
    inputs = keras.Input(shape=(32, 32, 6))
    num_filters = 64
    
    t = layers.BatchNormalization()(inputs)
    t = layers.Conv2D(kernel_size=3,
               strides=1,
               filters=num_filters,
               padding="same")(t)
    t = relu_bn(t)
    
    num_blocks_list = [2, 5, 5, 2]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            t = residual_block(t, downsample=(j==0 and i!=0), filters=num_filters)
        num_filters *= 2
    
    t = layers.AveragePooling2D(4)(t)
    t = layers.Flatten()(t)
    outputs_k = layers.Dense(k_dim, activation='softmax')(t)
    outputs_e = layers.Dense(1)(t)
    
    model = keras.Model(inputs, [outputs_k, outputs_e])

    return model

# The model to take the error, direction and z value and output the two inputs for the generator
direction_input = keras.Input(shape=(k_dim))
error_input = keras.Input(shape=(1))
z_input = keras.Input(shape=(latent_dim))
pre_a = layers.Multiply()([direction_input, error_input])
post_a = layers.Dense(latent_dim, use_bias=False, kernel_constraint = UnitNorm(axis = 1))(pre_a)
perturbed_out = layers.Add()([post_a, z_input])
model_A = keras.Model([direction_input, error_input, z_input], perturbed_out)
model_A.summary()

# The model or the reconstructor which takes two images, concataned in axis 3
reconstructor = create_res_net()

#%%
class DimensionDiscover(keras.Model):
    def __init__(self, model_A, decoder, model_R, **kwargs):
        super(DimensionDiscover, self).__init__(**kwargs)
        self.decoder = decoder
        self.model_A = model_A
        self.model_R = model_R

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
        directions, err, z = data
        with tf.GradientTape() as tape_r, tf.GradientTape() as tape_a:
            pert_out = self.model_A((directions, err, z))
            reconstruction = decoder(z)
            reconstruction_pert = decoder(pert_out)
            r_input = tf.concat([reconstruction, reconstruction_pert], axis = 3)
            out_k, out_e = self.model_R(r_input)
            k_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(directions, out_k)
            )
            e_loss = tf.reduce_mean(
                keras.losses.MeanAbsoluteError()(err, out_e)
            )
            total_loss = k_loss + e_loss * 0.25

            grads_r = tape_r.gradient(total_loss, self.model_R.trainable_weights)
            grads_a = tape_a.gradient(total_loss, self.model_A.trainable_weights)

            self.optimizer[0].apply_gradients(zip(grads_r, self.model_R.trainable_variables))
            self.optimizer[1].apply_gradients(zip(grads_a, self.model_A.trainable_variables))
        return {
            "loss": total_loss,
            "dimension_loss": k_loss,
            "noise_loss": e_loss,
        }

# %%

possible_dims = np.eye(k_dim)

def data_generator(k_dim, latent_dim, batch_size=None):

    while True:
        dims = possible_dims[np.random.choice(k_dim, size=batch_size)]
        errors = np.random.uniform(-6,6,(batch_size))
        z_vals = np.random.normal(0,1,(batch_size,latent_dim))

        yield dims, errors, z_vals

##TODO: change to arguments
batch_size = args.batch_size

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.load_weights(Path(args.weights_vae, 'weights.ckpt'))
dd = DimensionDiscover(model_A, vae.decoder, reconstructor)
dd.compile(optimizer=[keras.optimizers.Adam(learning_rate=0.0001), keras.optimizers.Adam(learning_rate=0.0001)])
dd.fit(data_generator(k_dim, latent_dim, batch_size=batch_size),
        epochs=100,
        steps_per_epoch = 512,
        batch_size=batch_size,
        callbacks=[cp_callback, es_callback])

# %%

if args.logs_dir:
    sys.stdout.close()
