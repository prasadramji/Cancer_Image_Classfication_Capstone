import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import resnet_model as cm


st.title("Cancer Image Classification")

st.markdown("\nsome text")

image_1 = st.file_uploader('Upload An Image')

predict_button = st.button('Predict on uploaded files', on_click=None)
test_data = st.button('Predict on sample data', on_click=None)

@st.cache
def create_model():
    resmodel = cm.create_model()
    return resmodel


def predict(image_1,resmodel,predict_button = predict_button):
    start = time.process_time()
    if predict_button:
        if (image_1 is not None):
            start = time.process_time()  
            image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
            image_1 = np.array(image_1)/255
            st.image([image_1],width=300)
            caption = cm.predict_Category([image_1],resmodel)
            st.markdown(" ### **Impression:**")
            impression = st.empty()
            impression.write(caption[0])
            time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
            st.write(time_taken)
            del image_1
        else:
            st.markdown("## Upload an Image")

def predict_sample(resmodel,folder = './test_images'):
    no_files = len(os.listdir(folder))
    file = np.random.randint(1,no_files)
    file_path = os.path.join(folder,str(file))
    image_1 = os.path.join(file_path,os.listdir(file_path)[0])
    predict(image_1,resmodel, True)
    

resmodel = create_model()


if test_data:
    predict_sample(resmodel)
else:
    predict(image_1,resmodel)
