#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# 데이터 로드 함수 정의
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

# U-Net 모델 정의
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    def conv_block(inputs, filters):
        x = Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        x = Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    def encoder_block(inputs, filters):
        x = conv_block(inputs, filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(inputs, skip_features, filters):
        x = Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        x = concatenate([x, skip_features])
        x = conv_block(x, filters)
        return x

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d4 = decoder_block(b, s4, 512)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    outputs = Conv2D(3, 1, activation='linear')(d1)

    model = Model(inputs, outputs, name='U-Net')
    return model

root_dir = 'GOPRO_Large/train'
max_images = 50

train_dataset = load_image_pairs(root_dir, max_images).batch(2)

model = unet_model()
model.compile(optimizer='adam', loss='mse')

model.fit(train_dataset, epochs=5)

model.save('UNET_model_0530.keras')




import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 업로드한 이미지 로드 및 전처리
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 후처리 함수
def postprocess_image(image_array):
    image_array = np.clip(image_array, 0, 1) * 255
    return image_array.astype(np.uint8)

# 모델 로드
model = load_model('UNET_model_0530.keras')

# 예측할 이미지 경로
image_path = 'image_path'

input_image = preprocess_image(image_path)

predicted_image = model.predict(input_image)

input_image_rescaled = postprocess_image(input_image[0])
predicted_image_rescaled = postprocess_image(predicted_image[0])



# 결과 시각화
plt.figure(figsize=(10, 5))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.title('Blurred Image')
plt.imshow(array_to_img(input_image[0]))

# 샤프해진 이미지
plt.subplot(1, 2, 2)
plt.title('Sharpened Image')
plt.imshow(array_to_img(predicted_image[0]))

plt.show()




import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 데이터 로드 함수 정의
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

    return np.array(blur_images), np.array(sharp_images)

# PSNR 및 SSIM 계산 함수
@tf.function
def predict_image(model, blur_image):
    return model(blur_image, training=False)

def evaluate_performance(model, blur_images, sharp_images):
    psnr_values = []
    ssim_values = []

    for i in range(len(blur_images)):
        blur_image = np.expand_dims(blur_images[i], axis=0)
        sharp_image = sharp_images[i]
        
        predicted_image = predict_image(model, blur_image)[0].numpy()
        predicted_image = np.clip(predicted_image, 0, 1)  # 모델의 출력을 [0, 1] 범위로 클리핑
        
        psnr_value = peak_signal_noise_ratio(sharp_image, predicted_image, data_range=1.0)
        
        # 이미지 크기에 맞게 win_size를 설정
        win_size = min(sharp_image.shape[0], sharp_image.shape[1], 7)
        ssim_value = structural_similarity(sharp_image, predicted_image, multichannel=True, data_range=1.0, win_size=win_size, channel_axis=-1)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
    
    return np.mean(psnr_values), np.mean(ssim_values)

# 데이터셋 경로
test_root_dir = 'GOPRO_Large/test'
max_images = 20

blur_images_test, sharp_images_test = load_image_pairs(test_root_dir, max_images)

loaded_model = load_model('UNET_model_0530.keras')

mean_psnr, mean_ssim = evaluate_performance(loaded_model, blur_images_test, sharp_images_test)

print(f'Mean PSNR: {mean_psnr}')
print(f'Mean SSIM: {mean_ssim}')

