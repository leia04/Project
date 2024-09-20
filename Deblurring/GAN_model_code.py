#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Add, Input, Conv2DTranspose, LeakyReLU, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16

def load_image_pairs(root_dir, max_images=None):
    blur_images = []
    sharp_images = []

    blur_count = 0
    sharp_count = 0

    for subdir, dirs, files in os.walk(root_dir):
        if 'blur' in subdir:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=(256, 256))
                    img = img_to_array(img) / 255.0
                    blur_images.append(img)
                    blur_count += 1
                    if max_images and blur_count >= max_images:
                        break
        elif 'sharp' in subdir:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=(256, 256))
                    img = img_to_array(img) / 255.0
                    sharp_images.append(img)
                    sharp_count += 1
                    if max_images and sharp_count >= max_images:
                        break

        if max_images and (blur_count >= max_images and sharp_count >= max_images):
            break

    min_count = min(len(blur_images), len(sharp_images))
    blur_images = blur_images[:min_count]
    sharp_images = sharp_images[:min_count]

    return tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(blur_images), tf.convert_to_tensor(sharp_images)))

root_dir = 'GOPRO_Large/train'
max_images = 50

train_dataset = load_image_pairs(root_dir, max_images).batch(8)

for blur, sharp in train_dataset.take(1):
    print("Blur images shape:", blur.shape)
    print("Sharp images shape:", sharp.shape)

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [w_pad, w_pad], [h_pad, h_pad], [0, 0]], 'REFLECT')

def generator_model():
    inputs = Input(shape=(256, 256, 3))

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=(7,7), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=64*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    mult = 2**n_downsampling
    for i in range(9):
        x = res_block(x, 64*mult, use_dropout=True)

    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = Conv2DTranspose(filters=int(64 * mult / 2), kernel_size=(3,3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = ReflectionPadding2D((3,3))(x)
    x = Conv2D(filters=3, kernel_size=(7,7), padding='valid')(x)
    x = Activation('tanh')(x)

    outputs = Add()([x, inputs])
    outputs = Lambda(lambda z: z/2)(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model

def discriminator_model():
    n_layers, use_sigmoid = 3, False
    inputs = Input(shape=(256, 256, 3))

    x = Conv2D(filters=64, kernel_size=(4,4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    nf_mult, nf_mult_prev = 1, 1
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=64*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    x = Conv2D(filters=64*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='tanh')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model

def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=(256, 256, 3))
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    model = Model(inputs=inputs, outputs=[generated_images, outputs])
    return model

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return tf.keras.backend.mean(tf.keras.backend.square(loss_model(y_true) - loss_model(y_pred)))

def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1), use_dropout=False):
    x = ReflectionPadding2D((1, 1))(input)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if use_dropout:
        x = Dropout(0.5)(x)

    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides)(x)
    x = BatchNormalization()(x)

    merged = Add()([input, x])
    return merged

def train(train_dataset, num_epochs):
    d_losses = []
    g_losses = []
    
    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)
        for step, (blur_images, sharp_images) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                generated_images = g(blur_images, training=True)

                d_real = d(sharp_images, training=True)
                d_fake = d(generated_images, training=True)

                d_loss_real = tf.reduce_mean(d_real)
                d_loss_fake = tf.reduce_mean(d_fake)

                d_loss = d_loss_fake - d_loss_real

            d_gradients = tape.gradient(d_loss, d.trainable_variables)
            d_opt.apply_gradients(zip(d_gradients, d.trainable_variables))

            with tf.GradientTape() as tape:
                generated_images = g(blur_images, training=True)

                d_fake = d(generated_images, training=True)
                g_loss_fake = -tf.reduce_mean(d_fake)

                p_loss = perceptual_loss(sharp_images, generated_images)

                g_loss = g_loss_fake + 0.01 * p_loss

            g_gradients = tape.gradient(g_loss, g.trainable_variables)
            g_opt.apply_gradients(zip(g_gradients, g.trainable_variables))

            d_losses.append(d_loss.numpy())
            g_losses.append(g_loss.numpy())

            print("Step:", step + 1, "D Loss:", d_loss.numpy(), "G Loss:", g_loss.numpy())

        save_images(epoch, g, train_dataset, batch_size, num_batches=3)
        
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='D Loss')
    plt.plot(g_losses, label='G Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_images(epoch, generator, dataset, batch_size, num_batches=1):
    batch_iter = iter(dataset)
    
    for batch_idx in range(num_batches):
        blur_images, sharp_images = next(batch_iter)
        gen_images = generator.predict(blur_images)

        num_images = min(batch_size, len(blur_images))

        fig, axs = plt.subplots(num_images, 3, figsize=(10, 10))
        for i in range(num_images):
            axs[i, 0].imshow(blur_images[i], vmin=0, vmax=1)
            axs[i, 0].set_title("Blurred")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(gen_images[i], vmin=0, vmax=1)
            axs[i, 1].set_title("Generated")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(sharp_images[i], vmin=0, vmax=1)
            axs[i, 2].set_title("Sharp")
            axs[i, 2].axis("off")

        plt.savefig(f"epoch_{epoch + 1}_batch_{batch_idx + 1}.png")
        plt.show()

g = generator_model()
d = discriminator_model()
d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

g_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
d_on_g_opt = Adam(learning_rate=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

d.trainable = True
d.compile(optimizer=d_opt, loss=wasserstein_loss)
d.trainable = False
loss = [perceptual_loss, wasserstein_loss]
loss_weights = [100, 1]
d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
d.trainable = True

epoch_num = 3
batch_size = 8

for epoch in range(epoch_num):
    print("Epoch:", epoch + 1)
    train(train_dataset, num_epochs=1)

g.save('GAN_model_final_2.h5')


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [w_pad, w_pad], [h_pad, h_pad], [0, 0]], 'REFLECT')

generator = load_model('GAN_model_final_2.h5', custom_objects={'ReflectionPadding2D': ReflectionPadding2D})

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  
    return img

def postprocess_image(image):
    image = image[0]  
    image = (image + 1) / 2.0  
    image = np.clip(image, 0, 1)  
    image = (image * 255).astype(np.uint8)  
    return image

def display_images(blur_image, predicted_image):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Blurred Image")
    plt.imshow(np.squeeze(blur_image, axis=0))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Sharp Image")
    predicted_image = postprocess_image(predicted_image)  
    plt.imshow(predicted_image)
    plt.axis('off')
    
    plt.show()

@tf.function
def predict_image(generator, blur_image):
    return generator(blur_image, training=False)

blur_image_path = 'KakaoTalk_Image_2024-05-30-23-06-49_004.jpeg'

blur_image = load_and_preprocess_image(blur_image_path)

predicted_image = predict_image(generator, blur_image)

display_images(blur_image, predicted_image)


# In[11]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_test_data(root_dir, max_images=None):
    blur_images = []
    sharp_images = []

    blur_count = 0
    sharp_count = 0

    for subdir, dirs, files in os.walk(root_dir):
        if 'blur' in subdir:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=(256, 256))
                    img = img_to_array(img) / 255.0
                    blur_images.append(img)
                    blur_count += 1
                    if max_images and blur_count >= max_images:
                        break
        elif 'sharp' in subdir:
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subdir, file)
                    img = load_img(img_path, target_size=(256, 256))
                    img = img_to_array(img) / 255.0
                    sharp_images.append(img)
                    sharp_count += 1
                    if max_images and sharp_count >= max_images:
                        break

        if max_images and (blur_count >= max_images and sharp_count >= max_images):
            break

    min_count = min(len(blur_images), len(sharp_images))
    blur_images = blur_images[:min_count]
    sharp_images = sharp_images[:min_count]

    return np.array(blur_images), np.array(sharp_images)

def evaluate_performance(generator, blur_images, sharp_images):
    psnr_values = []
    ssim_values = []

    for i in range(len(blur_images)):
        blur_image = np.expand_dims(blur_images[i], axis=0)
        sharp_image = sharp_images[i]
        
        predicted_image = generator.predict(blur_image)[0]
        predicted_image = (predicted_image + 1) / 2.0
        sharp_image = (sharp_image + 1) / 2.0
        
        psnr_value = peak_signal_noise_ratio(sharp_image, predicted_image, data_range=1.0)
        ssim_value = structural_similarity(sharp_image, predicted_image, multichannel=True, channel_axis=-1, data_range=1.0, win_size=11)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    
    return np.mean(psnr_values), np.mean(ssim_values)

test_root_dir = 'GOPRO_Large/test'
max_images = 20

blur_images, sharp_images = load_test_data(test_root_dir, max_images)

mean_psnr, mean_ssim = evaluate_performance(generator, blur_images, sharp_images)

print(f'Mean PSNR: {mean_psnr}')
print(f'Mean SSIM: {mean_ssim}')

