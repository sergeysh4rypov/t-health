import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- 1. Инициализация ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-3-flash-preview') 
else:
    st.error("⚠️ API Ключ Gemini не найден в .env")

# --- 2. Настройка страницы ---
st.set_page_config(page_title="Т-Health Pro AI", page_icon="🧬", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background: #262730; padding: 20px; border-radius: 15px; border-top: 4px solid #FFDD00; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Т-Health Pro: Глубокий биометрический анализ")
st.write("Версия 4.0: Улучшенная математика + VO2 Max и Глюкоза")

# --- 3. Генерация данных (Улучшенная нелинейная модель) ---
@st.cache_data
def load_health_data():
    np.random.seed(42)
    n = 3000
    actual_age = np.random.randint(18, 80, n)
    bmi = np.random.uniform(18, 40, n)
    hr_rest = np.random.randint(45, 110, n)
    sleep_hours = np.random.uniform(3, 11, n)
    stress = np.random.randint(1, 11, n)
    # Новые параметры:
    vo2_max = np.random.uniform(20, 60, n)    # Выносливость (чем выше, тем лучше)
    glucose = np.random.uniform(3.5, 8.0, n)  # Сахар (идеал 4.5-5.5)
    
    # --- УМНАЯ МАТЕМАТИКА ОМОЛОЖЕНИЯ ---
    bio_age = actual_age.astype(float)
    
    # ИМТ (штраф за отклонение от 22)
    bio_age += (bmi - 22)**2 * 0.12
    # Сон (штраф за недосып)
    bio_age += (8 - sleep_hours) * 2.8
    # Пульс (спортсмены омолаживаются)
    bio_age += (hr_rest - 60) * 0.15
    # VO2 Max (каждые 5 единиц выше нормы омолаживают на год)
    bio_age -= (vo2_max - 35) * 0.4
    # Глюкоза (резкий штраф за сахар > 5.5)
    bio_age += np.where(glucose > 5.5, (glucose - 5.5) * 5, 0)
    # Стресс
    bio_age += (stress - 3) * 1.5
    
    # Генетическая погрешность
    bio_age += np.random.normal(0, 1.5, n)
    
    return pd.DataFrame({
        'Возраст': actual_age, 'ИМТ': bmi, 'Пульс': hr_rest, 
        'Сон': sleep_hours, 'Стресс': stress, 'VO2_Max': vo2_max,
        'Глюкоза': glucose, 'Био_возраст': bio_age
    })

df = load_health_data()

# --- 4. Обучение модели ---
X = df.drop('Био_возраст', axis=1)
y = df['Био_возраст']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- 5. Интерфейс ввода (Sidebar) ---
with st.sidebar:
    st.header("📋 Лабораторные данные")
    in_age = st.number_input("Возраст (лет)", 18, 100, 20)
    
    st.subheader("Физиология")
    in_bmi = st.slider("ИМТ (вес/рост²)", 15.0, 45.0, 22.0)
    in_hr = st.slider("Пульс в покое", 40, 120, 60)
    in_vo2 = st.slider("VO2 Max (выносливость)", 15, 70, 45)
    
    st.subheader("Образ жизни и анализы")
    in_sleep = st.slider("Сон (часов)", 3.0, 12.0, 8.0)
    in_stress = st.select_slider("Стресс", options=list(range(1, 11)), value=2)
    in_glc = st.slider("Глюкоза натощак (ммоль/л)", 3.0, 10.0, 4.8)
    
    st.write("---")
    analyze_btn = st.button("🚀 Запустить PRO-анализ", use_container_width=True)

# --- 6. Основной блок ---
if analyze_btn:
    user_input = pd.DataFrame([[in_age, in_bmi, in_hr, in_sleep, in_stress, in_vo2, in_glc]], 
                              columns=X.columns)
    prediction = model.predict(user_input)[0]
    diff = prediction - in_age
    
    # 6.1 Метрики
    st.subheader("🏁 Биометрический отчет")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Биологический возраст", f"{prediction:.1f} лет", delta=f"{diff:.1f} лет", delta_color="inverse")
    with c2:
        msg = "Организм в отличном ресурсе! ✨" if diff < 0 else "Требуется корректировка образа жизни ⚠️"
        st.write(f"**Вердикт:** {msg}")
    with c3:
        percentile = (df['Био_возраст'] > prediction).mean() * 100
        st.metric("Здоровье лучше, чем у", f"{percentile:.1f}% популяции")

    # 6.2 Gemini Pro Аналитика
    st.write("---")
    st.subheader("🤖 Аналитика от Gemini 3 Flash")
    if api_key:
        with st.spinner('AI-агент изучает биомаркеры...'):
            prompt = f"""
            Ты — эксперт Т-Health. Проанализируй данные:
            Возраст: {in_age}, Био-возраст: {prediction:.1f}.
            Показатели: ИМТ {in_bmi}, Пульс {in_hr}, VO2Max {in_vo2}, Глюкоза {in_glc}, Сон {in_sleep}ч, Стресс {in_stress}/10.
            Оцени адекватность разрыва между паспортным и био-возрастом.
            Дай 3 профессиональных совета (по питанию, спорту и сну).
            Стиль: четко, по делу, с заботой.
            """
            try:
                response = llm_model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"Ошибка Gemini: {e}")

    # 6.3 Визуализации
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("📊 Влияние факторов")
        feat_df = pd.DataFrame({'Фактор': X.columns, 'Важность': model.feature_importances_}).sort_values('Важность')
        fig_imp = px.bar(feat_df, x='Важность', y='Фактор', orientation='h', color_discrete_sequence=['#FFDD00'], template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        st.subheader("📈 Распределение в базе")
        fig_dist = px.histogram(df, x="Био_возраст", nbins=50, color_discrete_sequence=['#444444'], template="plotly_dark")
        fig_dist.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="#FFDD00")
        st.plotly_chart(fig_dist, use_container_width=True)

else:
    st.info("Введите свои показатели слева для проведения глубокого анализа.")