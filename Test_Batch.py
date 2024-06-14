import os
import cv2
import numpy as np
from keras.api.models import load_model
from keras.api.applications.vgg16 import preprocess_input
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the saved model
loaded_model = load_model("VGG16_gender_identify_D-0.5_e-12_bs-3.h5")

# Function to load and preprocess test images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Set the path to the test image folder
test_folder = "D:\\Full AIML Course\\Gen AIML Tasks\\Gender_Identification\\Test\\Male"
f_count = 0
m_count = 0
female_pred = []
all_pred = []
# Iterate over the images in the test folder
for filename in os.listdir(test_folder):
    image_path = os.path.join(test_folder, filename)
    
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Perform prediction using the loaded model
    prediction = loaded_model.predict(img)
    
    # Get the predicted label
    if prediction[0] > 0.5:
        label = "Female"
        f_count += 1
        all_pred.append('Female')
    else:
        label = "Male"
        m_count += 1
        all_pred.append('Male')

    # Print the prediction result
    print(f"Image: {filename}, Prediction: {label}")
    
# Counting the number male and female predictions
print("F_Count: ", f_count)
print("M_Count: ", m_count)

# Confusion Matrix
# all_true = np.array(['Female']*100)
all_true = np.array(['Male']*101)

# Create the confusion matrix
# cm = confusion_matrix(all_true,all_pred,labels=["Female", "Male"])
cm = confusion_matrix(all_true, all_pred, labels=["Male", "Female"])

# Display the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Female', 'Male'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Male', 'Female'])
disp.plot(cmap='Blues')

plt.show()