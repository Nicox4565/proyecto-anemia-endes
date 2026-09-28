import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página
st.set_page_config(page_title="Triaje Predictivo", page_icon="🩸", layout="centered")

st.title("🩺 Aplicativo Ensamblado: Riesgo de Anemia")
st.write("Esta herramienta utiliza un **Ensamble Algorítmico (Votación Suave)** que combina el análisis estadístico de una Regresión Logística con el reconocimiento geométrico de fenotipos del K-Nearest Neighbors.")

# 2. Cargar los modelos
try:
    # Cargamos el archivo del ensamble que contiene a ambos modelos dentro
    ensamble = joblib.load('modelo_ensamblado.pkl')
    escalador = joblib.load('escalador_anemia.pkl')
except Exception as e:
    st.error("⚠️ Error: No se encontraron los archivos del modelo. Sube 'modelo_ensamblado.pkl' y 'escalador_anemia.pkl'")
    st.stop()

# 3. Interfaz de ingreso de datos
st.markdown("### 📝 Datos Clínicos de Triaje")
col1, col2 = st.columns(2)

with col1:
    peso = st.number_input("Peso (Kg)", min_value=30.0, max_value=150.0, value=60.0, step=0.1)
    talla = st.number_input("Talla (cm)", min_value=120.0, max_value=200.0, value=155.0, step=1.0)

with col2:
    embarazo = st.selectbox("¿Se encuentra embarazada?", ["No", "Sí"])
    seguro = st.selectbox("¿Cuenta con Seguro de Salud?", ["Sí", "No"])

# 4. Cálculos internos
bmi = peso / ((talla / 100) ** 2)
val_embarazo = 1 if embarazo == "Sí" else 0
val_seguro = 1 if seguro == "Sí" else 0

st.info(f"ℹ️ Índice de Masa Corporal (IMC) calculado: **{bmi:.2f}**")

# 5. Ejecución del Ensamble
if st.button("Calcular Riesgo Integrado", type="primary", use_container_width=True):
    
    # Preparamos los datos con el formato exacto del entrenamiento
    datos_paciente = pd.DataFrame(
        [[peso, talla, bmi, val_embarazo, val_seguro]], 
        columns=['Peso_Kg', 'Talla_cm', 'BMI', 'Embarazada', 'Tiene_Seguro']
    )
    
    datos_escalados = escalador.transform(datos_paciente)
    
    # EXTRAEMOS LAS PREDICCIONES INDIVIDUALES (La magia del Ensamble)
    # Extraemos qué opina la Regresión Logística
    prob_logreg = ensamble.named_estimators_['LogReg'].predict_proba(datos_escalados)[0][1]
    # Extraemos qué opina el KNN
    prob_knn = ensamble.named_estimators_['KNN'].predict_proba(datos_escalados)[0][1]
    # Extraemos el voto combinado final
    prob_final = ensamble.predict_proba(datos_escalados)[0][1]
    
    # 6. Mostrar Resultados
    st.markdown("---")
    st.subheader("📊 Desglose de la Predicción Algorítmica")
    
    # Mostramos los tres porcentajes en tarjetas alineadas
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Voto: Regresión Logística", f"{prob_logreg:.1%}", delta="Análisis de Variables", delta_color="off")
    col_b.metric("Voto: KNN", f"{prob_knn:.1%}", delta="Similitud de Pacientes", delta_color="off")
    col_c.metric("Predicción Final (Promedio)", f"{prob_final:.1%}", delta="Ensamble", delta_color="normal")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Umbral de decisión ajustado (40% para priorizar Recall en salud pública)
    umbral_clinico = 0.40 
    
    if prob_final >= umbral_clinico:
        st.error(f"⚠️ **RIESGO ALTO:** El modelo ensamblado determina una probabilidad predictiva de anemia del **{prob_final:.1%}**.")
        st.write("📌 **Recomendación Clínica:** Derivar de forma prioritaria a laboratorio para extracción de sangre y tamizaje de hemoglobina.")
    else:
        st.success(f"✅ **RIESGO BAJO:** El modelo ensamblado determina una probabilidad predictiva de anemia del **{prob_final:.1%}**.")
        st.write("📌 **Recomendación Clínica:** Atención regular de triaje. No requiere prioridad urgente en laboratorio.")

st.markdown("---")
st.caption("Consideración Ética: Este modelo tecnológico utiliza Machine Learning para investigación académica (Design Science Research) y no sustituye el diagnóstico hematológico definitivo.")