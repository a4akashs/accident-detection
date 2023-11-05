from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Data loading and preprocessing code (modify paths accordingly)
# Assuming you have a "train" directory with "accident" and "non-accident" subfolders

# Set the path to your "train" directory
train_data_dir = 'C:/Users/santh/OneDrive/Desktop/accident detection/train'

# Create data generators
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split data into training and validation
)

# Load and preprocess data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Adjust the target size as needed
    batch_size=32,  # Adjust batch size as needed
    class_mode='binary',  # 'binary' for binary classification
    subset='training'  # 'training' for the training subset
)

validation_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output (1 neuron)
])

# Model Compilation
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(train_generator,
                    epochs=10,  # Adjust the number of training epochs
                    batch_size=32,  # Adjust batch size
                    validation_data=validation_generator)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(validation_generator)  # Assuming you want to evaluate on the validation data
print(f"Validation Accuracy: {test_accuracy}")

# Save the Model
model.save('accident_detection_model.h5')
