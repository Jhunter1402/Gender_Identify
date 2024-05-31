import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt
from keras.api.models import load_model
from keras.api.applications.vgg16 import preprocess_input

# Load the saved model
loaded_model = load_model("gender_identify_D-0_e-20_bs-12.h5")

# Function to load and preprocess test images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to Predict Gender
def predict_gender(image_path):
    img = preprocess_image(image_path)
    prediction = loaded_model.predict(img)

    # Assuming your model outputs a single probability for the gender
    gender = 'Female' if prediction[0][0] > 0.5 else 'Male'
    return gender
def main():
    # Create a simple Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected")
        return

    # Predict the gender of the selected image
    gender = predict_gender(file_path)
    print(f'The detected gender is: {gender}')

    # Load the image for plotting
    img = Image.open(file_path)

    # Plot the image with the gender label
    plt.imshow(img)
    plt.title(f'Predicted Gender: {gender}')
    plt.axis('off')  # Hide the axis
    plt.show()


if __name__ == '__main__':
    main()
