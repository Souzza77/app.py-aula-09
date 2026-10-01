import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Detector YOLO", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Detecção de Objetos em Tempo Real")
st.sidebar.header("Configurações")
confidence = st.sidebar.slider("Limiar de Confiança", 0.1, 1.0, 0.5, 0.05)

option = st.sidebar.selectbox("Fonte de Entrada", ["Imagem", "Vídeo"])

if option == "Imagem":
    uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        results = model.predict(source=image, conf=confidence)
        res_plotted = results[0].plot()
        
        st.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), caption="Resultado da Detecção", use_container_width=True)
        
        del file_bytes, image, results, res_plotted

elif option == "Vídeo":
    uploaded_video = st.file_uploader("Envie um vídeo", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
        
        cap = cv2.VideoCapture("temp_video.mp4")
        stframe = st.empty()
        
        stop_btn = st.button("Parar Processamento")
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(source=frame, conf=confidence, verbose=False)
            res_plotted = results[0].plot()
            
            stframe.image(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            del frame, results, res_plotted
            
        cap.release()