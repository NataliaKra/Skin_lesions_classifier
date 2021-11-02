import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()

st.title('Skin Lesions Classifier')

st.markdown("Welcome to Skin Lesions Classifier - the app that helps to recognise cancerous and non-cancerous lesions")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["jpg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', width=150)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "best_CNN_Datagen_2cl_categ.hdf5"
    model = load_model(classifier_model)
    test_image = image.resize((120,120))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Non-Cancerous',
          'Cancerous'
          ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Non-Cancerous': 0,
          'Cancerous': 0
}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result
 

if __name__ == "__main__":
    main()