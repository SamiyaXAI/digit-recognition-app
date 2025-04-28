import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# Load the trained CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
model.eval()

# Streamlit App Title
st.title('🖌️ Handwritten Digit Recognition App')

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Prediction button
if st.button('Predict'):
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28,28))
        img = img / 255.0
        img = 1 - img
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            prediction = torch.argmax(output, dim=1).item()
        
        st.success(f'**Predicted Digit: {prediction}**')
    else:
        st.warning("Please draw a digit first!")
