import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)

model = tf.keras.models.load_model('saved_models/best_model.h5')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = 20
batch_size = 64

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=[
              tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
              tf.keras.callbacks.ModelCheckpoint(filepath='saved_models/best_model_augmented.h5', save_best_only=True)
          ])
