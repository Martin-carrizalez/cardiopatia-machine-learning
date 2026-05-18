# Predicción de Enfermedad Cardíaca mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

## 📋 Descripción

Proyecto final de la materia **Introducción a la IA: Machine Learning** del Centro Universitario de Guadalajara, Universidad de Guadalajara.

Se aplicaron técnicas de Machine Learning sobre el dataset UCI Heart Disease Cleveland (Janosi et al., 1989) con el objetivo de identificar patrones clínicos asociados a la presencia de enfermedad cardíaca.

## 🚀 Dashboard interactivo

> **[Ver dashboard en Streamlit Cloud](URL_AQUI)**

## 📊 Modelos implementados

| Modelo | Accuracy | AUC |
|---|---|---|
| Árbol de Decisión (podado) | 72% | 0.75 |
| SVM | 87% | 0.78 |
| Red Neuronal (MLP) | 85% | 0.93 |
| K-Means | Clustering | — |

La **Red Neuronal** obtuvo el mejor AUC (0.93), siendo el modelo más recomendable para contexto médico donde detectar falsos negativos tiene mayor costo que una falsa alarma.

## 🗂️ Estructura del repositorio

```
cardiopatia-machine-learning/
│
├── notebook.ipynb          # Análisis completo: EDA, preprocesamiento y modelos
├── app.py                  # Dashboard interactivo en Streamlit
├── requirements.txt        # Dependencias del proyecto
│
├── modelo_arbol.pkl        # Árbol de Decisión entrenado
├── modelo_svm.pkl          # SVM entrenado
├── modelo_red.pkl          # Red Neuronal entrenada
├── modelo_kmeans.pkl       # K-Means entrenado
├── estandarizador.pkl      # StandardScaler ajustado
├── resultados.pkl          # Métricas de evaluación
├── X_test.pkl              # Conjunto de prueba (features)
├── X_test_scaled.pkl       # Conjunto de prueba estandarizado
└── y_test.pkl              # Conjunto de prueba (target)
```

## 🔬 Dataset

**UCI Heart Disease — Cleveland**
- 303 pacientes, 13 variables clínicas
- Variable objetivo binarizada: 0 = Sano, 1 = Enfermedad cardíaca
- Fuente: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

## ⚙️ Instalación local

```bash
git clone https://github.com/TU_USUARIO/cardiopatia-machine-learning
cd cardiopatia-machine-learning
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Variables más importantes

Según el análisis de importancia del Árbol de Decisión:

1. **Tipo de dolor de pecho** (`cp`) — variable más determinante
2. **Vasos principales coloreados** (`ca`)
3. **Talasemia** (`thal`)
4. **Depresión del segmento ST** (`oldpeak`)

## 👨‍🎓 Información académica

- **Alumno:** Martin Angel Carrizalez Piña
- **Profesor:** Juan Carlos López
- **Materia:** Introducción a la IA: Machine Learning
- **Institución:** Centro Universitario de Guadalajara, Universidad de Guadalajara
- **Fecha:** Mayo 2026
