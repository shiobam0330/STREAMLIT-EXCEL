import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import predict_model
import tempfile

path = "D:/Downloads/INTELIGENCIA ARTIFICIAL/ACTIVIDAD VALENTINA hechaaa/STREAMLIT/"
with open(path+'best_model.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

st.title("Predicción del precio basado en el uso del usuario")
archivo = st.file_uploader("Cargar archivo Excel", type=["xlsx", "csv"])

if st.button("Predecir"):
    if archivo is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(archivo.read())
                tmp_path = tmp_file.name
            
            if archivo.name.endswith(".csv"):
                prueba = pd.read_csv(tmp_path,header = 0,sep=";",decimal=",")
            else:
                prueba = pd.read_excel(tmp_path)
            
            covariables = ['Avg. Session Length', 'Time on App', 'Time on Website',
                           'Length of Membership', 'dominio', 'Tec']
            base = prueba.get(covariables)
            prediccion = predict_model(dt2, data=base)
            predicciones = pd.DataFrame({'Email': prueba["Email"],
                                      'Precio': prediccion["prediction_label"]})
            st.write("Predicciones generadas correctamente!")
            st.write(predicciones)

            st.download_button(label="Descargar archivo de predicciones",
                               data=predicciones.to_csv(index=False),
                               file_name="Predicciones.csv",
                               mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Por favor, cargue un archivo válido.")

if st.button("Reiniciar"):
    st.experimental_rerun()