from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.layers.merge import add
from keras.models import Sequential, Model
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras import backend as K
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import pickle

def half_mse(y_true, y_pred):
    return 0.5 * K.mean(K.square(y_pred - y_true), axis=-1)



class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.d_losses = []
        self.g_losses = []
        self.epoch = 0

        # Configure data loader
        self.dataset_name = 'vangogh2photo'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.validation_weight = 1.0
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 5.0    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss=half_mse,
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])
        self.d_B.compile(loss=half_mse,
            optimizer=Adam(0.0002, 0.5),
            metrics=['accuracy'])

        # Build and compile the generators

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  self.validation_weight, self.validation_weight,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=Adam(0.0002, 0.5))

    

    def build_generator(self):
        """U-Net Generator"""

        def c7s1_k(y, k, final):
            y = Conv2D(k, kernel_size=(7,7), strides=1, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization()(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = Activation('relu')(y)
            return y

        def d_k(y,k):
            y = Conv2D(k, kernel_size=(3,3), strides=2, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization()(y)
            y = Activation('relu')(y)
            return y

        def R_k(y, k):
            shortcut = y

            # down-sampling is performed with a stride of 2
            y = Conv2D(k, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization()(y)
            y = Activation('relu')(y)
            
            y = Conv2D(k, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization()(y)
            y = Activation('relu')(y)

            return add([shortcut, y])

        def u_k(y,k):
            y = Conv2DTranspose(k, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer = self.weight_init)(y)
            y = InstanceNormalization()(y)
            y = Activation('relu')(y)
            
            return y


        # Image input
        d0 = Input(shape=self.img_shape)

        y = c7s1_k(d0, 64, False)
        y = d_k(y, 128)
        y = d_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = R_k(y, 256)
        y = u_k(y, 128)
        y = u_k(y, 64)
        y = c7s1_k(y, 3, True)
        output_img = y

   
        return Model(d0, output_img)

    def build_discriminator(self):

        def C_k(y,k, norm=True):
            y = Conv2D(k, kernel_size=(4,4), strides=2, padding='same', kernel_initializer = self.weight_init)(y)
            if norm:
                y = InstanceNormalization()(y)
            y = LeakyReLU(0.2)(y)
            return y

        img = Input(shape=self.img_shape)

        y = C_k(img, 64, False)
        y = C_k(y, 128)
        y = C_k(y, 256)
        y = C_k(y, 512)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same',kernel_initializer = self.weight_init)(y)

        return Model(img, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(self.epoch, epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

                d_loss = [
                    d_loss_total[0]
                    , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
                    , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
                    , d_loss_total[1]
                    , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
                    , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
                ]


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    % ( self.epoch, epochs,
                        batch_i, self.data_loader.n_batches,
                        d_loss_total[0], 100*d_loss_total[1],
                        g_loss[0],
                        np.sum(g_loss[1:3]),
                        np.sum(g_loss[3:5]),
                        np.sum(g_loss[5:7]),
                        elapsed_time))

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(batch_i)
                    self.save_model()

                
            self.epoch += 1

    def sample_images(self, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 4

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        # imgs_A = self.data_loader.load_img('datasets/%s/testA/n07740461_14740.jpg' % self.dataset_name)
        # imgs_B = self.data_loader.load_img('datasets/%s/testB/n07749192_4241.jpg' % self.dataset_name)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        # ID the images
        id_A = self.g_BA.predict(imgs_A)
        id_B = self.g_AB.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.clip(gen_imgs, 0, 1)

        titles = ['Original', 'Translated', 'Reconstructed', 'ID']
        fig, axs = plt.subplots(r, c, figsize=(15,7.5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, self.epoch, batch_i))
        plt.close()

    def save_model(self):

        os.makedirs('saved_model/%s' % self.dataset_name, exist_ok=True)

        self.combined.save_weights('saved_model/%s/model-%d.h5' % (self.dataset_name, self.epoch))
        self.d_A.save_weights('saved_model/%s/d_A-%d.h5' % (self.dataset_name, self.epoch))
        self.d_B.save_weights('saved_model/%s/d_B-%d.h5' % (self.dataset_name, self.epoch))
        self.g_BA.save_weights('saved_model/%s/g_BA-%d.h5' % (self.dataset_name, self.epoch))
        self.g_AB.save_weights('saved_model/%s/g_AB-%d.h5' % (self.dataset_name, self.epoch))
                
        pickle.dump(self, open( "saved_model/%s/obj.pkl" % (self.dataset_name), "wb" ))


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1, sample_interval=100)
