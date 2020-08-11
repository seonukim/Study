import tensorflow as tf
import torch

class CycleGAN():
    def __init__(self):
        # 입력 이미지 shape
        self.img_rows = 128     # 이미지의 가로 픽셀
        self.img_cols = 128     # 이미지의 세로 픽셀
        self.channels = 3       # 이미지 채널
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # data loader 설정
        self.dataset_name = 'apple2orange'  # 데이터셋 디렉토리 이름
        self.data_loader = torch.utils.data.DataLoader(dataset_name = self.dataset_name,
                                                       img_res = (self.img_rows, self.img_cols))
        
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

