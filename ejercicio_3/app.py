import streamlit as st
import pickle
import numpy as np
import lightgbm as lgb
# Crear inputs para cada atributo
import pandas as pd

@st.cache_resource
def load_model():
    with open('modelo_smote_calibrado.pkl', 'rb') as f:
        modelo = pickle.load(f)
        
    return modelo

modelo = load_model()

st.title("Predicciones de fallas utilizando LightGBM")
st.write("Ingresa los valores de los atributos para obtener una predicción")


# Agregar los 9 atributos del modelo
atributos = {
    'attribute1': st.number_input("Atributo 1", value=48467332.0),
    'attribute2': st.number_input("Atributo 2", value=64776.0),
    'attribute3': st.number_input("Atributo 3", value=0.0),
    'attribute4': st.number_input("Atributo 4", value=841.0), 
    'attribute5': st.number_input("Atributo 5", value=8.0),
    'attribute6': st.number_input("Atributo 6", value=39267.0),
    'attribute7': st.number_input("Atributo 7", value=56.0),
    'attribute9': st.number_input("Atributo 9", value=1.0)
}

if st.button("Predecir"):
    # Convertir los atributos a un DataFrame
    print("atributos: ", atributos)
    X = pd.DataFrame([atributos], columns=atributos.keys())
    
    # Realizar predicción
    probabilidad_falla = modelo.predict_proba(X)[0][1]
    print("probabilidad_falla: ", probabilidad_falla)
    clasificacion = "Falla!" if probabilidad_falla > 0.0515 else "No Falla!"
    print("clasificacion: ", clasificacion)
    # Mostrar resultados
    if clasificacion == "Falla!":
        st.error(f"Clasificación: {clasificacion}")
    else:
        st.success(f"Clasificación: {clasificacion}")
    st.info(f"Probabilidad: {probabilidad_falla:.8f}")
    
    # Gráfico de confianza
    st.progress(float(probabilidad_falla))