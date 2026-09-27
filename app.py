import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Triaje Predictivo - Anemia", page_icon="🩸", layout="centered")

st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Logotipo_del_Ministerio_de_Salud_del_Per%C3%BA.png/800px-Logotipo_del_Ministerio_de_Salud_del_Per%C3%BA.png", width=200)
st.title("🩺 Sistema de Triaje: Predicción de Anemia")
st.write("Ingrese los datos de la paciente para evaluar la probabilidad de riesgo predictivo basado en el modelo de la ENDES 2024.")

# 2. Cargar los modelos generados en Colab
try:
    modelo = joblib.load('modelo_logistico.pkl')
    escalador = joblib.load('escalador_anemia.pkl')
except Exception as e:
    st.error("⚠️ Error: No se encontraron los archivos del modelo (.pkl). Asegúrate de que estén en el mismo repositorio.")
    st.stop()

# 3. Interfaz de ingreso de datos
st.markdown("### 📝 Datos de la Paciente")
col1, col2 = st.columns(2)

with col1:
    peso = st.number_input("Peso (Kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.1)
    talla = st.number_input("Talla (cm)", min_value=120.0, max_value=200.0, value=155.0, step=1.0)

with col2:
    embarazo = st.selectbox("¿Se encuentra embarazada?", ["No", "Sí"])
    seguro = st.selectbox("¿Cuenta con Seguro de Salud?", ["Sí", "No"])

# 4. Cálculos internos
# Calculamos el IMC matemáticamente (BMI)
bmi = peso / ((talla / 100) ** 2)

# Convertimos las respuestas a números (1 = Sí, 0 = No) tal como lo aprendió la IA
val_embarazo = 1 if embarazo == "Sí" else 0
val_seguro = 1 if seguro == "Sí" else 0

st.info(f"ℹ️ Índice de Masa Corporal (IMC) calculado: **{bmi:.2f}**")

# 5. Botón de Predicción
if st.button("Calcular Riesgo de Anemia", type="primary", use_container_width=True):
    
    # Armamos el dataframe exactamente igual que en Colab
    datos_paciente = pd.DataFrame(
        [[peso, talla, bmi, val_embarazo, val_seguro]], 
        columns=['Peso_Kg', 'Talla_cm', 'BMI', 'Embarazada_1', 'Tiene_Seguro_1']
    )
    
    # Estandarizamos los datos
    datos_escalados = escalador.transform(datos_paciente)
    
    # Predicción de la probabilidad (Clase 1 = Con Anemia)
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    
    # 6. Resultados
    st.markdown("---")
    st.subheader("📊 Resultado del Análisis Predictivo")
    
    if probabilidad >= 0.5:
        st.error(f"⚠️ **RIESGO ALTO:** La paciente tiene un **{probabilidad:.1%}** de probabilidad algorítmica de presentar anemia.")
        st.write("📌 **Recomendación Clínica:** Derivar de forma prioritaria a laboratorio para extracción de sangre y tamizaje de hemoglobina.")
    else:
        st.success(f"✅ **RIESGO BAJO:** La paciente tiene solo un **{probabilidad:.1%}** de probabilidad algorítmica de presentar anemia.")
        st.write("📌 **Recomendación Clínica:** Atención regular de triaje. No requiere prioridad urgente en laboratorio.")

st.markdown("---")
st.caption("Consideración Ética: Este sistema es un artefacto tecnológico de apoyo desarrollado para fines académicos (Design Science Research) y no sustituye un diagnóstico médico definitivo.")