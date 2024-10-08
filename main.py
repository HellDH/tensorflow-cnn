import os

import keras

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from keras import Sequential, layers

train_path = os.getcwd()

batch_size = 32
img_width = 256
img_height = 256

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

class_names = train_ds.class_names

print(f"Class names: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = Sequential([
	layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

	layers.Conv2D(16, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(32, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Conv2D(64, 3, padding='same', activation='relu'),
	layers.MaxPooling2D(),

	layers.Flatten(),
	layers.Dense(128, activation='relu'),
	layers.Dense(num_classes)
])

model.compile(
	optimizer='adam',
	loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy'])

model.summary()

epochs = 6
history = model.fit(
	train_ds,
	validation_data=val_ds,
	epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

fruit_path = r"C:\Users\cybersport\Desktop\Utkin IS 23.9\testimgs\test.jpg"

img = keras.utils.load_img(
    fruit_path, target_size=(img_height, img_width)
)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(f"На изображении скорее всего {class_names[np.argmax(score)]} с вероятностью {100 * np.max(score):.2f}%")

img.show()

model.save("model", save_format="h5")