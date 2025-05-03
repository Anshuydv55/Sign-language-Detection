import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('custom_cnn_best_weights.keras')

# Define image dimensions
img_width, img_height = 64, 64

# Define 29 class labels: A-Z + SPACE, DELETE, NOTHING
class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
}

# Initialize the GUI window
top = tk.Tk()
top.geometry('800x600')
top.title('Sign Language Detector')
top.configure(background='#CDCDCD')

# Create labels
label1 = Label(top, background="#CDCDCD", font=('arial', 20, "bold"))
sign_image = Label(top)

# Function to process the image and predict the sign
def Detect(file_path):
    try:
        # Load and preprocess the image
        img = Image.open(file_path).convert('L')  # Grayscale
        img = img.resize((img_width, img_height))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # (64, 64, 1)
        img_array = np.expand_dims(img_array, axis=0)   # (1, 64, 64, 1)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_sign = class_labels.get(predicted_index, "Unknown")

        print(f"Predicted Sign: {predicted_sign}")
        label1.configure(foreground="#011638", text=predicted_sign)

    except Exception as e:
        print("Error in detection:", e)

# Function to show Detect button
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Sign", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Function to upload an image
def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        uploaded = Image.open(file_path)
        uploaded.thumbnail((350, 350))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')

        show_Detect_button(file_path)

    except Exception as e:
        print("Error:", e)

# Upload button
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

# Place labels
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)

# Heading
heading = Label(top, text="Sign Language Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Run the GUI
top.mainloop()
