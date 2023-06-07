import streamlit as st
import io
import os
from model import load_model
from predict import predict, labels

from PIL import Image

# Model of CNN
MODEL_PATH = os.path.join("result_trainModel", "ClassificationEyeDisease.pth")

# For load uploaded image

def load_image(file_upload):
    if file_upload is not None:
        image_data = file_upload.getvalue()
        st.image(image_data, width=320)
        return Image.open(io.BytesIO(image_data))
    else:
        return None    

# Dashboard
def main():
    st.title("Eye Disease Classification")
    st.write("Detect 4 Class : Cataract, Diabetic Retinopathy, Glaucoma, Normal")

    with st.sidebar:
        st.title("Upload image")
        file_upload = st.file_uploader(label="Eye Disease Classfication : ", type=['png', 'jpg', 'jpeg'])

        st.title("Reach Me")
        st.write("LinkedIn : [Hardianto Tandi Seno](https://www.linkedin.com/in/hardianto-ts/)")
        st.write("Gmail : [hardiantotandiseno@gmail.com](https://mail.google.com/mail/?view=cm&to=hardiantotandiseno@gmail.com&su=SUBJECT&body=BODY)")

    col1, col2 = st.columns(2)

    model = load_model(MODEL_PATH)
    with col1:
        st.write("Image will show in here")
        image = load_image(file_upload)

    with col2:
        result = st.button('Run on image')
        if result:
            st.write("Result & Accuracy of Prediction : ")
            for i, (prob,label) in enumerate(predict(model, image)):
                st.write(f"{i+1}. {labels[label]} ({prob*100:.2f})")

                st.markdown(
                    """
                    <style>
                    .stProgress .st-b8{
                        background-color: orange;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(prob)

if __name__ == '__main__':
    main()

