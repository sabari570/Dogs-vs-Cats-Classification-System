from io import StringIO, BytesIO
import streamlit as st
import pickle
from PIL import Image
import numpy as np

model = pickle.load(open('trained_model.pkl', 'rb'))

st.title('Dog vs Cat Predictor')

uploaded_file = st.file_uploader("Upload the image in the form of '.jpg'", type=[".jpg", "png", ".jpeg"])
show_file = st.empty()
if not uploaded_file:
    show_file.info("Please upload a file of type: " + " ".join([".jpg", ".png", ".jpeg"]))

if isinstance(uploaded_file, BytesIO):
    file_name = uploaded_file.name
    file_path = '/home/sabari/Downloads/'+file_name
    img = Image.open(file_path)
    st.image(img)
    img_resized = img.resize((224, 224))
    img_array = np.asarray(img_resized)
    img_scaled = img_array/255
    img_reshaped = np.reshape(img_scaled, [1, 224, 224, 3])
    prediction = model.predict(img_reshaped)
    final_prediction = np.argmax(prediction)
    if final_prediction == 1:
        st.success("The image represents a dog")
    else:
        st.success("The image represents a cat")

