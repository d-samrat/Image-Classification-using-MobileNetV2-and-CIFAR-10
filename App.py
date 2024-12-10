import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

def mobilenetv2_imagenet():
    st.title("MobileNetV2 for Image Classification")
    uploaded_file = st.file_uploader("Select an image", type=['jpg','png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        st.write('Classifying the image...')

        model = tf.keras.applications.MobileNetV2(weights='imagenet')

        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score*100:.2f}%")

def cifar10_classification():
    st.title("CIFAR-10 for Image Classification")

    uploaded_file = st.file_uploader("Select an image", type=["jpg","png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)

        st.write("Classifying the image...")

        model = tf.keras.models.load_model('model111.h5')

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        img = image.resize((32,32))
        img_array = np.array(img)
        img_array = img_array.astype('float32')/255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.write(f"Predicted class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence*100:.2f}%")

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose a model", ("CIFAR-10", "MobileNetV2"))

    if choice=='MobileNetV2':
        mobilenetv2_imagenet()
    elif choice=='CIFAR-10':
        cifar10_classification()

if __name__ == "__main__":
    main()