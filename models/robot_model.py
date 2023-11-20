import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, ReLU, Concatenate, Softmax
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Dropout


def get_model():
    # Image input path (64x64 image with 3 color channels)
    image_input = Input(shape=(64, 64, 3), name='image_input')
    # Convolutional layers with increasing filters but not as many as in the deeper model
    x_image = Conv2D(filters=8, kernel_size=(4, 4), activation='relu', padding='same')(image_input)
    x_image = Conv2D(filters=16, kernel_size=(4, 4), activation='relu', padding='same')(x_image)
    x_image = Flatten()(x_image)

    # LiDAR input path (20-dimensional input)
    lidar_input = Input(shape=(20,), name='lidar_input')
    x_lidar = Dense(64, activation='relu')(lidar_input)
    x_lidar = Dense(64, activation='relu')(x_lidar)

    # Concatenate the outputs of the two networks
    x = Concatenate()([x_image, x_lidar])

    # A few more Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # Output layer with Softmax
    output = Dense(3, activation='softmax')(x)

    # Create the model
    model = Model(inputs=[image_input, lidar_input], outputs=output)

    # Compile the model
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
