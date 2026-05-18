import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

# ── Cargar modelos ──────────────────────────────────────────────────────────
arbol = joblib.load('modelo_arbol.pkl')
svm = joblib.load('modelo_svm.pkl')
red = joblib.load('modelo_red.pkl')
kmeans = joblib.load('modelo_kmeans.pkl')
estandarizador = joblib.load('estandarizador.pkl')
X_test = joblib.load('X_test.pkl')
X_test_scaled = joblib.load('X_test_scaled.pkl')
y_test = joblib.load('y_test.pkl')

# ── Traducciones ────────────────────────────────────────────────────────────
NOMBRES_VARIABLES = {
    'age': 'Edad',
    'sex': 'Sexo',
    'cp': 'Tipo de dolor de pecho',
    'trestbps': 'Presión arterial en reposo (mmHg)',
    'chol': 'Colesterol sérico (mg/dl)',
    'fbs': 'Glucosa en ayunas > 120 mg/dl',
    'restecg': 'Electrocardiograma en reposo',
    'thalach': 'Frecuencia cardíaca máxima',
    'exang': 'Angina por ejercicio',
    'oldpeak': 'Depresión del segmento ST',
    'slope': 'Pendiente del ST en ejercicio',
    'ca': 'Vasos principales coloreados',
    'thal': 'Talasemia',
    'target': 'Diagnóstico'
}

ETIQUETAS_CATEGORICAS = {
    'sex': {0: 'Mujer', 1: 'Hombre'},
    'cp': {1: 'Típico', 2: 'Atípico', 3: 'No anginoso', 4: 'Asintomático'},
    'fbs': {0: 'Normal', 1: 'Alta'},
    'restecg': {0: 'Normal', 1: 'Anormalidad ST-T', 2: 'Hipertrofia'},
    'exang': {0: 'No', 1: 'Sí'},
    'slope': {1: 'Ascendente', 2: 'Plana', 3: 'Descendente'},
    'ca': {0: '0 vasos', 1: '1 vaso', 2: '2 vasos', 3: '3 vasos'},
    'thal': {3: 'Normal', 6: 'Defecto fijo', 7: 'Defecto reversible'},
    'target': {0: 'Sano', 1: 'Enfermo'}
}

# ── Cargar datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    heart = fetch_ucirepo(id=45)
    df = pd.DataFrame(heart.data.features)
    df['target'] = heart.data.targets
    df['ca'] = df['ca'].fillna(df['ca'].median())
    df['thal'] = df['thal'].fillna(df['thal'].median())
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = cargar_datos()
columnas_escalar = ['chol', 'trestbps', 'thalach', 'oldpeak']
df_scaled = df.copy()
df_scaled[columnas_escalar] = estandarizador.transform(df_scaled[columnas_escalar])
X = df.drop(columns=['target'])
y = df['target']
X_scaled = df_scaled.drop(columns=['target'])
y_scaled = df_scaled['target']

# ── Configuración página ────────────────────────────────────────────────────
st.set_page_config(page_title="Dashboard Cardíaco", layout="wide")
st.title("Dashboard — Predicción de Enfermedad Cardíaca")
st.caption("Dataset UCI Heart Disease Cleveland — Janosi et al. (1989)")

seccion = st.sidebar.radio("Navegación", [
    "Resumen del Dataset",
    "Métricas de Modelos",
    "Curva ROC",
    "Predictor de Riesgo",
    "Clustering K-Means"
])

# ── Sección 1: Resumen ──────────────────────────────────────────────────────
if seccion == "Resumen del Dataset":
    st.header("Resumen del Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total pacientes", len(df))
    col2.metric("Pacientes sanos", int((y == 0).sum()))
    col3.metric("Pacientes enfermos", int((y == 1).sum()))

    st.subheader("Distribución del diagnóstico")
    df_diag = df.copy()
    df_diag['Diagnóstico'] = df_diag['target'].map(ETIQUETAS_CATEGORICAS['target'])
    fig = px.histogram(df_diag, x='Diagnóstico', color='Diagnóstico',
                       color_discrete_map={'Sano': 'steelblue', 'Enfermo': 'crimson'})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explorar distribución por variable")
    variable = st.selectbox(
        "Selecciona una variable",
        options=list(X.columns),
        format_func=lambda x: NOMBRES_VARIABLES.get(x, x)
    )

    df_var = df.copy()
    df_var['Diagnóstico'] = df_var['target'].map(ETIQUETAS_CATEGORICAS['target'])

    if variable in ETIQUETAS_CATEGORICAS:
        df_var[variable] = df_var[variable].map(ETIQUETAS_CATEGORICAS[variable])
        fig2 = px.histogram(df_var, x=variable, color='Diagnóstico',
                            barmode='group',
                            color_discrete_map={'Sano': 'steelblue', 'Enfermo': 'crimson'},
                            title=f"Distribución de {NOMBRES_VARIABLES[variable]} por diagnóstico",
                            labels={variable: NOMBRES_VARIABLES[variable]})
    else:
        fig2 = px.box(df_var, x='Diagnóstico', y=variable, color='Diagnóstico',
                      color_discrete_map={'Sano': 'steelblue', 'Enfermo': 'crimson'},
                      title=f"Distribución de {NOMBRES_VARIABLES[variable]} por diagnóstico",
                      labels={variable: NOMBRES_VARIABLES[variable]})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Mapa de correlaciones")
    corr = df.corr()
    corr.index = [NOMBRES_VARIABLES.get(c, c) for c in corr.index]
    corr.columns = [NOMBRES_VARIABLES.get(c, c) for c in corr.columns]
    fig3 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
    st.plotly_chart(fig3, use_container_width=True)

# ── Sección 2: Métricas ─────────────────────────────────────────────────────
elif seccion == "Métricas de Modelos":
    st.header("Comparativa de Modelos")

    from sklearn.model_selection import cross_val_score

    resultados = joblib.load('resultados.pkl')

    col1, col2, col3 = st.columns(3)
    col1.metric("Árbol Podado — Accuracy", f"{resultados['Árbol Podado']:.1%}")
    col2.metric("SVM — Accuracy", f"{resultados['SVM']:.1%}")
    col3.metric("Red Neuronal — Accuracy", f"{resultados['Red Neuronal']:.1%}")

    fig = px.bar(
    x=['Árbol Podado', 'SVM', 'Red Neuronal'],
    y=[resultados['Árbol Podado'], resultados['SVM'], resultados['Red Neuronal']],
    labels={'x': 'Modelo', 'y': 'Accuracy'},
    color=['Árbol Podado', 'SVM', 'Red Neuronal'],
    title="Accuracy por modelo (conjunto de prueba)"
    )

    fig.add_hline(y=0.8, line_dash='dash', line_color='red', annotation_text='Umbral 80%')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Importancia de Variables — Árbol de Decisión")
    importancias = pd.Series(arbol.feature_importances_, index=X.columns).sort_values()
    importancias.index = [NOMBRES_VARIABLES.get(c, c) for c in importancias.index]
    fig2 = px.bar(importancias, orientation='h',
                  labels={'value': 'Importancia', 'index': 'Variable'},
                  title="Importancia de variables — Árbol de Decisión")
    st.plotly_chart(fig2, use_container_width=True)

    from sklearn.metrics import confusion_matrix

    st.subheader("Matrices de Confusión")

    modelos_cm = {
        'Árbol Podado': (arbol.predict(X_test), y_test),
        'SVM': (svm.predict(X_test_scaled), y_test),
        'Red Neuronal': (red.predict(X_test_scaled), y_test)
    }

    col1, col2, col3 = st.columns(3)
    for col, (nombre, (y_pred, y_real)) in zip([col1, col2, col3], modelos_cm.items()):
        cm = confusion_matrix(y_real, y_pred)
        fig_cm = px.imshow(cm,
                        text_auto=True,
                        labels=dict(x='Predicción', y='Real'),
                        x=['Sano', 'Enfermo'],
                        y=['Sano', 'Enfermo'],
                        color_continuous_scale='Blues',
                        title=nombre)
        col.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Análisis de Falsos Negativos por Modelo")
    st.caption("Pacientes enfermos que el modelo clasificó como sanos — los casos más peligrosos")

    X_test_display = X_test.copy()
    X_test_display.columns = [NOMBRES_VARIABLES.get(c, c) for c in X_test_display.columns]
    X_test_display['Diagnóstico Real'] = y_test.values

    for nombre, y_pred in [
        ('Árbol Podado', arbol.predict(X_test)),
        ('SVM', svm.predict(X_test_scaled)),
        ('Red Neuronal', red.predict(X_test_scaled))
    ]:
        falsos_negativos = X_test_display[(y_test.values == 1) & (y_pred == 0)]
        st.markdown(f"**{nombre} — {len(falsos_negativos)} falsos negativos**")
        if len(falsos_negativos) > 0:
            st.dataframe(falsos_negativos.drop(columns=['Diagnóstico Real']))
        else:
            st.success("Sin falsos negativos")

# ── Sección 3: Curva ROC ────────────────────────────────────────────────────
elif seccion == "Curva ROC":
    st.header("Curva ROC — Comparativa de Modelos")
    st.caption("Calculada sobre el conjunto de prueba (20% de los datos)")

    prob_arbol = arbol.predict_proba(X_test)[:, 1]
    prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
    prob_red = red.predict_proba(X_test_scaled)[:, 1]

    fpr_a, tpr_a, _ = roc_curve(y_test, prob_arbol)
    fpr_s, tpr_s, _ = roc_curve(y_test, prob_svm)
    fpr_r, tpr_r, _ = roc_curve(y_test, prob_red)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_a, y=tpr_a, name=f'Árbol Podado (AUC={auc(fpr_a,tpr_a):.2f})'))
    fig.add_trace(go.Scatter(x=fpr_s, y=tpr_s, name=f'SVM (AUC={auc(fpr_s,tpr_s):.2f})'))
    fig.add_trace(go.Scatter(x=fpr_r, y=tpr_r, name=f'Red Neuronal (AUC={auc(fpr_r,tpr_r):.2f})'))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Azar'))
    fig.update_layout(
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        title='Curva ROC — Conjunto de prueba'
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Sección 4: Predictor ────────────────────────────────────────────────────
elif seccion == "Predictor de Riesgo":
    st.header("Predictor de Riesgo Cardíaco")
    st.markdown("Ingresa los datos clínicos del paciente:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Edad", 20, 100, 54)
        sex = st.selectbox("Sexo", [0, 1], format_func=lambda x: ETIQUETAS_CATEGORICAS['sex'][x])
        cp = st.selectbox("Tipo de dolor de pecho", [1, 2, 3, 4], format_func=lambda x: ETIQUETAS_CATEGORICAS['cp'][x])
        trestbps = st.number_input("Presión arterial en reposo (mmHg)", 80, 220, 130)
        chol = st.number_input("Colesterol sérico (mg/dl)", 100, 600, 246)
    with col2:
        fbs = st.selectbox("Glucosa en ayunas > 120 mg/dl", [0, 1], format_func=lambda x: ETIQUETAS_CATEGORICAS['fbs'][x])
        restecg = st.selectbox("Electrocardiograma en reposo", [0, 1, 2], format_func=lambda x: ETIQUETAS_CATEGORICAS['restecg'][x])
        thalach = st.number_input("Frecuencia cardíaca máxima", 60, 220, 150)
        exang = st.selectbox("Angina por ejercicio", [0, 1], format_func=lambda x: ETIQUETAS_CATEGORICAS['exang'][x])
        oldpeak = st.number_input("Depresión del segmento ST", 0.0, 7.0, 1.0)
    with col3:
        slope = st.selectbox("Pendiente del ST en ejercicio", [1, 2, 3], format_func=lambda x: ETIQUETAS_CATEGORICAS['slope'][x])
        ca = st.selectbox("Vasos principales coloreados", [0, 1, 2, 3], format_func=lambda x: ETIQUETAS_CATEGORICAS['ca'][x])
        thal = st.selectbox("Talasemia", [3, 6, 7], format_func=lambda x: ETIQUETAS_CATEGORICAS['thal'][x])
        modelo_elegido = st.selectbox("Modelo a usar", ["SVM", "Red Neuronal", "Árbol Podado"])

    if st.button("Predecir", type="primary"):
        datos = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        df_input = pd.DataFrame(datos, columns=X.columns)
        df_input_scaled = df_input.copy()
        df_input_scaled[columnas_escalar] = estandarizador.transform(df_input_scaled[columnas_escalar])

        if modelo_elegido == "SVM":
            pred = svm.predict(df_input_scaled)[0]
            prob = svm.predict_proba(df_input_scaled)[0][1]
        elif modelo_elegido == "Red Neuronal":
            pred = red.predict(df_input_scaled)[0]
            prob = red.predict_proba(df_input_scaled)[0][1]
        else:
            pred = arbol.predict(df_input)[0]
            prob = arbol.predict_proba(df_input)[0][1]

        st.metric("Probabilidad de enfermedad cardíaca", f"{prob:.1%}")
        if pred == 1:
            st.error("⚠️ El modelo detecta riesgo de enfermedad cardíaca")
        else:
            st.success("✅ El modelo no detecta riesgo de enfermedad cardíaca")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Nivel de riesgo (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "crimson" if pred == 1 else "steelblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

# ── Sección 5: K-Means 3D ───────────────────────────────────────────────────
elif seccion == "Clustering K-Means":
    st.header("Clustering K-Means")

    clusters = kmeans.labels_

    st.subheader("Visualización 3D (PCA 3 componentes)")
    pca3 = PCA(n_components=3)
    X_pca3 = pca3.fit_transform(X_scaled)
    centroides_pca3 = pca3.transform(kmeans.cluster_centers_)

    df_3d = pd.DataFrame(X_pca3, columns=['Componente 1', 'Componente 2', 'Componente 3'])
    df_3d['Cluster'] = pd.Series(clusters.astype(str)).map({'0': 'Grupo 0', '1': 'Grupo 1'}).values
    df_3d['Diagnóstico Real'] = pd.Series(y_scaled.values.astype(str)).map({'0': 'Sano', '1': 'Enfermo'}).values

    fig3d = px.scatter_3d(df_3d,
                      x='Componente 1', y='Componente 2', z='Componente 3',
                      color='Cluster',
                      title="K-Means — Vista 3D (PCA)",
                      color_discrete_sequence=['steelblue', 'red'],
                      opacity=0.7)

    fig3d.add_scatter3d(
        x=centroides_pca3[:, 0],
        y=centroides_pca3[:, 1],
        z=centroides_pca3[:, 2],
        mode='markers',
        marker=dict(symbol='diamond', size=8, color='yellow',
                    line=dict(color='black', width=2)),
        name='Centroides'
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.subheader("Distribución clusters vs diagnóstico real")
    tabla = pd.crosstab(clusters, y_scaled,
                        rownames=['Cluster'],
                        colnames=['Diagnóstico Real (0=Sano, 1=Enfermo)'])
    st.dataframe(tabla)