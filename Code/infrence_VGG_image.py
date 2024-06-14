from PIL import Image
import numpy
from keras.models import load_model

l_model = load_model(r'C:\Desktop\POC\vgg16_reloaded_1516_17.h5')
#dictionary to label all traffic signs class.
classes = { 
    0:'Non-Defected Image',
    1:'Defected Image',
 
}

def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((224,224))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image/255
    pred = l_model.predict([image])
    print(pred)
    if pred>0.5:
        sign=classes[1]
    else:
        sign=classes[0]
    print(sign)

file_path = r'C:\Users\Desktop\POC\T1.jpg'  
classify(file_path)