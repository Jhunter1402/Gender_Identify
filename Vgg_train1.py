import os
import cv2
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.api.applications.vgg16 import VGG16
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten

# Function to load carton box images from a folder
def gender_images_from_folder(folder, label, subset_size):
    images = []
    labels = []
    for filename in os.listdir(folder)[:subset_size]:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            images.append(img)
            labels.append(label)
    return images, labels

# Set the number of samples per class for the subset (limited dataset)
subset_size = 101

# Load defective and defect-free carton box images
female_images, female_labels = gender_images_from_folder("D:\\Full AIML Course\\Gen AIML Tasks\\Gender_Identification\\gender\\train\\female", 1, subset_size)
male_images, male_labels = gender_images_from_folder("D:\\Full AIML Course\\Gen AIML Tasks\\Gender_Identification\\gender\\train\\male", 0, subset_size)

# Combine defective and defect-free images and labels
images = female_images[:subset_size] + male_images[:subset_size]
labels = female_labels[:subset_size] + male_labels[:subset_size]

# Convert images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Normalize images
images = images / 255.0

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained base model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=12)

# Evaluate the model on the training set
loss, accuracy = model.evaluate(X_train, y_train)
print('Training Loss:', loss)
print('Training Accuracy:', accuracy)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(X_val, y_val)
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

# Save the model to a file
model.save("gender_identify_D-0.5_e-20_bs-12.h5")
