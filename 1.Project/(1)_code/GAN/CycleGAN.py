import torch
import datetime
import numpy as np
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

class CycleGAN():
    def __init__(self):
        # 입력 이미지 shape
        self.img_rows = 128     # 이미지의 가로 픽셀
        self.img_cols = 128     # 이미지의 세로 픽셀
        self.channels = 3       # 이미지 채널
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # data loader 설정
        self.dataset_name = 'apple2orange'  # 데이터셋 디렉토리 이름
        self.data_loader = torch.utils.data.DataLoader(dataset_name=self.dataset_name,
                                                       img_res=(self.img_rows, self.img_cols))
        
        # Discriminator output 사이즈 계산(PatchGAN 사용)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Generator와 Discriminator의 첫 번째 레이어 필터 갯수
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0        # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identify loss

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        # Discriminators를 Build, Compile
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.d_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # -----------------------
        # Construct Computational
        # -----------------------

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # 각 도메인의 이미지를 입력합니다. Keras의 Input을 사용
        img_A = tf.keras.layers.Input(shape=self.img_shape)
        img_B = tf.keras.layers.Input(shape=self.img_shape)

        # 각각의 서로 다른 도메인으로 이미지를 바꿔줌
        # 사과, 오렌지 도메인이 있다고 할 때 사과는 오렌지처럼, 오렌지는 사과처럼 바꿔준다
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # 이미지의 원래 도메인으로 다시 바꿔줌
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identify Loss를 위해
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # generator를 학습할 때는 discriminator는 학습하지 않습니다
        self.d_A.trainable = False
        self.d_B.trainable = False

        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        self.combined = tf.keras.models.Model(
            inputs=[img_A, img_B],
            outputs=[valid_A, valid_B,
                     reconstr_A, reconstr_B,
                     img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1, 1, self.lambda_cycle, self.lambda_id, self.lambda_id],
                              optimizer=optimizer)
    
    def build_generator(self):
        '''U-Net Generator'''

        def conv2d(layer_input, filters, f_size=4):
            '''Downsampling 하는 레이어'''
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d
        
        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            '''Upsampling 하는 레이어'''
            u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
            u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = tf.keras.layers.Dropout(rate=dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = tf.keras.layers.Concatenate()([u, skip_input])
            return u
        
        # 이미지 입력. Keras의 Input을 사용
        d0 = tf.keras.layers.Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = tf.keras.layers.UpSampling2D(size=2)(u3)
        output_img = tf.keras.layers.Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return tf.keras.models.Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            '''Discriminator layer'''
            d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
            d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d
        
        img = tf.keras.layers.Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return tf.keras.models.Model(img, validity)
                  
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # --------------------
                # Train Discriminators
                # --------------------

                # 반대의 도메인으로 이미지 translate
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # discriminators 학습
                # (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ----------------
                # Train Generators
                # ----------------

                # generators 학습
                 g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                       [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imbs_A, imbs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss:%f, acc:%3d%%] [G loss:%05f, adv:%05f, recon:%05f, id:%05f] time:%s"\
                    % (epoch, epochs,
                       batch_i, self.data_loader.n_batches,
                       d_loss[0], 100*d_loss[1],
                       g_loss[0],
                       np.mean(g_loss[1:3]),
                       np.mean(g_loss[3:5]),
                       np.mean(g_loss[5:6]),
                       elapsed_time))
                
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_imgaes(epoch, batch_i)