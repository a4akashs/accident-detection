from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0, 1]
    validation_split=0.2  # Split data into training and validation
)

# Set the path to your "train" directory
train_data_dir = 'C:/Users/santh/OneDrive/Desktop/accident detection/train'

# Load and preprocess data
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Adjust the target size as needed
    batch_size=32,  # Adjust batch size as needed
    class_mode='binary',  # 'binary' for binary classification
    subset='training'  # 'training' for the training subset
)
