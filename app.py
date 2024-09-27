import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objs as go
import plotly.figure_factory as ff
import joblib
import os
import time

# Sabitler
MODEL_PATH = "lung_cancer_model_{}.joblib"
SCALER_PATH = "scaler.joblib"
DATA_PATH = "lung_cancer.xlsx"

# Tema ve stil ayarları
st.set_page_config(page_title="Akciğer Kanseri Tahmin Uygulaması", page_icon="🫁", layout="wide")

# CSS stilleri
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #262730
    }
    .Widget>label {
        color: #262730;
        font-family: sans-serif;
    }
    .stTextInput>div>div>input {
        color: #262730;
    }
    .stSelectbox>div>div>select {
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_excel(DATA_PATH)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

@st.cache_resource
def train_and_save_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }

    trained_models = {}
    accuracies = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        trained_models[name] = model
        accuracies[name] = accuracy
        joblib.dump(model, MODEL_PATH.format(name.lower().replace(' ', '_')))

    joblib.dump(scaler, SCALER_PATH)

    return trained_models, scaler, accuracies, X_test, y_test

@st.cache_resource
def load_models_and_scaler():
    models = {}
    for name in ['Random Forest', 'Logistic Regression', 'SVM', 'Decision Tree']:
        model_path = MODEL_PATH.format(name.lower().replace(' ', '_'))
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
    
    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    
    return models, scaler

def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    elif hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
    else:
        return None

    fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h',
        marker_color='rgba(50, 171, 96, 0.7)',
    ))
    fig.update_layout(
        title={
            'text': "Özellik Önemliliği",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Önem Derecesi",
        yaxis_title="Özellikler",
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def main():
    st.title('Akciğer Kanseri Tahmin Uygulaması')
    st.markdown("---")

    df = load_and_preprocess_data()
    if df is None:
        return

    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    models, scaler = load_models_and_scaler()
    if not models or scaler is None:
        with st.spinner('Modeller eğitiliyor, lütfen bekleyin...'):
            models, scaler, accuracies, X_test, y_test = train_and_save_models(X, y)
        st.success(f"Modeller başarıyla eğitildi.")

    
    selected_model = st.sidebar.selectbox("Model Seçin", list(models.keys()))
    model = models[selected_model]

    variables = {
        'Cinsiyet': 'GENDER',
        'Yaş': 'AGE',
        'Sigara İçme': 'SMOKING',
        'Sarı Parmaklar': 'YELLOW_FINGERS',
        'Anksiyete': 'ANXIETY',
        'Akran Baskısı': 'PEER_PRESSURE',
        'Kronik Hastalık': 'CHRONIC DISEASE',
        'Yorgunluk': 'FATIGUE ',
        'Alerji': 'ALLERGY ',
        'Hırıltılı Solunum': 'WHEEZING',
        'Alkol Tüketimi': 'ALCOHOL CONSUMING',
        'Öksürük': 'COUGHING',
        'Nefes Darlığı': 'SHORTNESS OF BREATH',
        'Yutma Zorluğu': 'SWALLOWING DIFFICULTY',
        'Göğüs Ağrısı': 'CHEST PAIN'
    }

    st.sidebar.title("Kullanıcı Bilgileri")
    user_input = {}
    
    for turkish, english in variables.items():
        if turkish == 'Yaş':           
            age = st.sidebar.number_input(f"{turkish}:", min_value=0, max_value=100, value=30)
            if age > 100:
                st.sidebar.warning("100 yaşın altında bir değer girin. Ölümsüz olmaya çalışmayın :)")
            user_input[english] = age
        elif turkish == 'Cinsiyet':
            gender = st.sidebar.selectbox(f"{turkish}:", ('Erkek', 'Kadın'))
            user_input[english] = 1 if gender == 'Erkek' else 0
        else:
            choice = st.sidebar.selectbox(f"{turkish}:", ('Hayır', 'Evet'))
            user_input[english] = 2 if choice == 'Evet' else 1

    if st.sidebar.button('Tahmin Et', key='predict'):
        with st.spinner("Tahmin yapılıyor..."):
            time.sleep(1)  # Simüle edilmiş işlem süresi
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probabilities = model.predict_proba(input_scaled)

        st.markdown("## Tahmin Sonuçları")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 1:
                st.error('Tahmin: Akciğer kanseri riski yüksek.')
            else:
                st.success('Tahmin: Akciğer kanseri riski düşük.')

            st.metric("Akciğer kanseri olma olasılığı", f"{probabilities[0][1]:.2%}")
            st.metric("Akciğer kanseri olmama olasılığı", f"{probabilities[0][0]:.2%}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probabilities[0][1],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Akciğer Kanseri Riski", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.5], 'color': 'cyan'},
                        {'range': [0.5, 0.7], 'color': 'royalblue'},
                        {'range': [0.7, 1], 'color': 'red'}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': probabilities[0][1]}}))
            fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
            st.plotly_chart(fig)

    st.markdown("---")
    st.markdown("## Model Analizi")
    
    tab1, tab2 = st.tabs(["📊 Özellik Önemliliği", "📈 Model Performansı"])
    
    with tab1:
        st.markdown("### Özellik Önemliliği")
        fig = plot_feature_importance(model, X.columns)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Bu model için özellik önemliliği gösterilemiyor.")
    
    with tab2:
        st.markdown("### Model Performans Metrikleri")
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sınıflandırma Raporu")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            
            # Check if df_report is empty before displaying
            if not df_report.empty:
                st.dataframe(df_report)
            else:
                st.warning("Sınıflandırma raporu oluşturulamadı. Modelin eğitildiğinden emin olun.")
    
        with col2:
            st.markdown("#### Karmaşıklık Matrisi")
            cm = confusion_matrix(y_test, y_pred)
            fig = ff.create_annotated_heatmap(
                z=cm, 
                x=['Negatif', 'Pozitif'], 
                y=['Negatif', 'Pozitif'],
                colorscale='Blues'
            )
            fig.update_layout(title='Karmaşıklık Matrisi', xaxis_title='Tahmin Edilen', yaxis_title='Gerçek Değer')
            st.plotly_chart(fig, use_container_width=True)

if _name_ == "_main_":
    main()
