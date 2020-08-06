import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
path = 'D:/Study/1.Project/(5)_Result'
os.chdir(path)

class DCGAN():
    def __init__(self, rows, cols, channels, z = 10):
        # input shape
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = z
        self.noise_shape = self.img_shape

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # 판별자 빌드 및 컴파일
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        
        # 생성자 빌드
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = tf.keras.layers.Input(shape=(self.noise_shape))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = tf.keras.models.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def build_generator(self):
        model = tf.keras.models.Sequential([
            # tf.keras.layers.Dense(128 * 7 * 7, input_dim=self.latent_dim, activation='relu'),
            # tf.keras.layers.Reshape((7, 7, 128)),
            # tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.Activation(activation='relu'),
            tf.keras.layers.Conv2D(self.channels, kernel_size=(3,3), padding='same'),
            tf.keras.layers.Activation(activation='tanh')
        ])

        model.summary()

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)
    
    def build_discriminator(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2),
                                   input_shape=self.img_shape, padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'),
            tf.keras.layers.ZeroPadding2D(padding=((0.1), (0.1))),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(momentum=0.8),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1),
            tf.keras.layers.Activation(activation='sigmoid')
        ])

        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)
    
    def train(self, epochs, batch_size=256, save_interval=50):

        # Load the dataset
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

        # Rescale -1 to 1
        x_train = x_train / 127.5 - 1.
        x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            '''
            Train Discriminator
            '''
            # Select a random half of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            '''
            Train Generator
            '''

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
    
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(path + 'dcgan_mnist_%d.png' % epoch)
        plt.close()

dcgan = DCGAN(28, 28, 1)
dcgan.train(epochs=5000, batch_size=256, save_interval=50)