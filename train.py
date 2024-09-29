import os

import keras

import numpy as np

import tensorflow as tf

batch_size = 32
img_width = 256
img_height = 256

train_path = os.getcwd()

train_ds = keras.utils.image_dataset_from_directory(
	train_path,
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(
	train_path,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

model = keras.models.load_model(r"C:\Users\cybersport\Desktop\Utkin IS 23.9\tensorflow_1\model")

class_names = train_ds.class_names

fruit_path = r"C:\Users\cybersport\Desktop\Utkin IS 23.9\testimgs\epl.jpg"

img = keras.utils.load_img(
    fruit_path, target_size=(img_height, img_width)
)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"На изображении скорее всего {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score):.2f}%")

img.show()