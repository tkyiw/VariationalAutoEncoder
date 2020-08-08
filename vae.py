import numpy as np
import glob
from PIL import Image

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Dropout, Activation, Flatten
from tensorflow.keras.layers import Reshape, Embedding, InputLayer
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau


# Arbitrary image data acquisition
path = 'image/train'
img_list = glob.glob(path + '/*' + ".png")
train_data = np.empty((0,320,896,3))
for img in img_list:
    img_ = Image.open(img)
    img = np.array(img_)
    img = img / 255.
    img = img.reshape((1,1000,1000,3))
    img = img[:, 340:660, 57:953, :]
    train_data = np.concatenate([train_data, img], axis = 0)

path = 'image/test/error'
img_list = glob.glob(path + '/*' + ".png")
error_data = np.empty((0,320,896,3))
for img in img_list:
    img_ = Image.open(img)
    img = np.array(img_)
    img = img / 255.
    img = img.reshape((1,1000,1000,3))
    img = img[:, 340:660, 57:953, :]
    error_data = np.concatenate([error_data, img], axis = 0)

path = 'image/test/good'
img_list = glob.glob(path + '/*' + ".png")
good_data = np.empty((0,320,896,3))
for img in img_list:
    img_ = Image.open(img)
    img = np.array(img_)
    img = img / 255.
    img = img.reshape((1,1000,1000,3))
    img = img[:, 340:660, 57:953, :]
    good_data = np.concatenate([good_data, img], axis = 0)

# reparameterization
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

input_shape = (320, 896, 3)
latent_dim = 20
K.clear_session()

# VAE model = encoder + decoder
# encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
shape = K.int_shape(x)
print("shape[1], shape[2], shape[3]",shape[1], shape[2], shape[3])
x = Flatten()(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
# decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# Compute VAE loss
reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                          K.flatten(outputs))
reconstruction_loss *= 320 * 896
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Running autoencoder
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=40,
                              min_lr=0.0001
                              )

hist = vae.fit(train_data,
               epochs=200,
               batch_size=5,
               validation_data=(error_data, None),
               callbacks=[reduce_lr],
               )

# save model
vae.save('vae.h5', include_optimizer=False)
