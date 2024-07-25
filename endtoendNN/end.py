import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GroupNormalization, Dense, Flatten
from tensorflow.keras.models import Sequential


model = Sequential([
    GroupNormalization(groups=1, axis=-1, input_shape=(66, 200, 3)),
    Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
    Conv2D(48, (3, 3), strides=(2, 2), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(1164, activation='relu'),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mean_absolute_error'])


image_batch = tf.keras.preprocessing.image_dataset_from_directory(
    placeholder,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(66, 200),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    data_format=None,
    verbose=True
)

model.fit(image_batch, epochs=50)

loss, mae = model.evaluate(image_batch)
print(f'Loss: {loss}')
print(f'Mean Absolute Error: {mae}')
