import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization

# Define paths and data generators
BS = 32  # Batch size
TS = (24, 24)  # Target image size
train_dir = 'C:\\Users\\BIT\\Downloads\\Drowsiness detection\\dataset_new\\train'
val_dir = 'C:\\Users\\BIT\\Downloads\\Drowsiness detection\\dataset_new\\test'

train_datagen = ImageDataGenerator(rescale=1/255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1/255)

# Load data and convert labels to categorical format
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=TS,
    batch_size=BS,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=TS,
    batch_size=BS,
    class_mode='categorical'
)

# Get number of steps per epoch for training and validation
SPE = len(train_generator.classes) // BS
VS = len(validation_generator.classes) // BS

# Uncomment and run the following lines to visualize sample images
# img, label = next(train_generator)
# plt.imshow(img[0])
# plt.title(f"Label: {label[0]}")


# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(TS[0], TS[1], 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=15, steps_per_epoch=SPE, validation_data=validation_generator, validation_steps=VS)

# Save the model
model.save('models/cnnCat2_augmented.keras', overwrite=True)
