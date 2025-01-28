import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scripts.download_data import download_dataset, split_dataset

# Paths for dataset
raw_data_path = "data/raw"
train_path = "data/train"
val_path = "data/val"
test_path = "data/test"

# Check if train, val, and test directories are missing, then download and split the dataset
if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
    print("Dataset not found. Downloading and preparing the dataset...")
    # Download the dataset (Update dataset name as per your requirement)
    download_dataset(dataset_name="username/dataset-name", data_dir=raw_data_path)
    # Split the dataset into train, val, and test
    split_dataset(source_dir=raw_data_path, train_dir=train_path, val_dir=val_path, test_dir=test_path)

# Define parameters
batch_size = 32
image_size = (200, 200)
img_channels = 3
n_classes = 36

# Image Data Generator for rescaling images
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow from Directory for training, validation, and test data
train_data = datagen.flow_from_directory(
    directory=train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    directory=val_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    directory=test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Model Architecture
model = Sequential([
    # First Convolution Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Second Convolution Block
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Third Convolution Block
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    # Fully Connected Layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')  # Output Layer
])

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
    verbose=0
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    factor=0.5,
    verbose=1
)

# Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary
model.summary()

# Fit the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stopping, reduce_learning_rate],
    verbose=1
)

# Save the Model
os.makedirs("models", exist_ok=True)
model.save('models/asl_model.h5')

# Evaluate the Model on Test Data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")