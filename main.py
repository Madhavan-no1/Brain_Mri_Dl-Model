import numpy as np
from PIL import Image

def getPrediction(filename, model):
    # Class names for brain tumor classification
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Resize image to the size that your model expects (150x150)
    SIZE = 150
    img = Image.open(filename).convert('RGB')
    img = img.resize((SIZE, SIZE))  # Resize the image to 150x150
    
    # Convert image to array and normalize pixel values
    img = np.asarray(img) / 255.0
    
    # Expand dimensions to match input format for the model (batch size, height, width, channels)
    img = np.expand_dims(img, axis=0)
    
    # Ensure the image shape is (1, 150, 150, 3)
    print("Image shape before feeding to model:", img.shape)
    
    # Make a prediction (this will output an array of probabilities or class indexes)
    pred = model.predict(img)
    
    # Convert prediction to integer (index of the class with the highest probability)
    predicted_class_index = np.argmax(pred, axis=1)[0]
    
    # Convert prediction to class name
    pred_class = classes[predicted_class_index]
    
    print(f"Diagnosis is: {pred_class}")
    return pred_class
