import streamlit as st
import tensorflow as tf
from streamlit_lottie import st_lottie
import requests
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)

def load_model():
    model = tf.keras.models.load_model('plantV12.hdf5')
    return model
model=load_model()
st.write(""" # Plant Leaf Disease Detection Using CNN """ )
st.write(""" ### Final Year Major Project """)
st.write(""" ### Made by: Shivam Sunil Bhosale (RA1711003010257), Shubham Pandey (RA1711003010003)""")
file= st.file_uploader("Upload a leaf image to make a prediction",type=['jpg',"png"])

import cv2 
from PIL import Image, ImageOps
import numpy as np
 
def import_and_predict(image_data,model):
    size = (227,227)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape= img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please Upload a leaf image file of size (256X256)")
else:
    image = Image.open(file)
    st.image(image,width=300)
    predictions = import_and_predict(image,model)
    class_names = ["Pepper bacterial spot","Pepper healthy","Tomato early blight","Tomato healthy"]
    string="The above leaf is most likely to be: "+class_names[np.argmax(predictions)]
    st.success(string) 
    
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()



