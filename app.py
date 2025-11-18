import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="Classificador CIFAR-10", layout="centered")
st.title("Classificador de Imagens — CIFAR-10")
st.write("Envie uma imagem para o modelo identificar a classe.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo_cifar.h5")
    return model

model = load_model()

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

uploaded = st.file_uploader("Envie uma imagem (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagem enviada", width=250)

    img_resized = ImageOps.fit(img, (32, 32), Image.ANTIALIAS)
    x = np.array(img_resized).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = np.argmax(preds)
    st.subheader(f"Predição: **{CLASSES[idx]}**")
    
    st.write("Probabilidades top-3:")
    top3 = np.argsort(preds[0])[-3:][::-1]
    for i in top3:
        st.write(f"{CLASSES[i]} — {preds[0][i]:.3f}")
