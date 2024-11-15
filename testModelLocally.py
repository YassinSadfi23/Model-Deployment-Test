import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

# Load the model
model_path = "classifier_resnet_model.h5"
model = tf.keras.models.load_model(model_path)

# Preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to the target size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img_array

# Make predictions
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Interpret the predictions
def interpret_predictions(predictions):
    class_labels = ["no crack", "crack"]
    predicted_class = np.argmax(predictions, axis=1)
    return class_labels[predicted_class[0]]

# Function to load and display the image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Make predictions and interpret the results
        predictions = predict_image(model, img_array)
        result = interpret_predictions(predictions)
        
        # Display the result
        result_label.config(text=f"The model predicts: {result}")

# Create the main window
root = tk.Tk()
root.title("Crack Detection Model")

# Create a button to load the image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack()

# Create a label to display the image
panel = Label(root)
panel.pack()

# Create a label to display the result
result_label = Label(root, text="")
result_label.pack()

# Run the application
root.mainloop()