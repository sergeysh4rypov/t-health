import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st
from supabase import create_client, Client

# Инициализация Supabase
try:
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
except:
    st.warning("База данных не подключена. Проверьте Secrets.")

# Блок кастомного дизайна
st.markdown("""
    <style>
    /* 1. Настройка фона всей страницы */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        background-attachment: fixed;
    }

    /* 2. Шрифты и основной текст */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #f8fafc;
    }

    /* 3. Эффект матового стекла для карточек (Glassmorphism) */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 10px;
    }

    /* 4. Стилизация заголовка */
    h1 {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }

    /* 5. Кнопка "Рассчитать" — еще более сочная */
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 1rem;
        font-size: 18px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }

    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.4);
        background: linear-gradient(90deg, #fbbf24, #f59e0b);
    }

    /* 6. Кастомные слайдеры */
    .stSlider [data-baseweb="slider"] {
        background-color: transparent;
    }
    
    /* Скрываем стандартный футер Streamlit */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
st.title("🧬 T-Health AI Pro")
st.markdown("<p style='text-align: center; color: #94a3b8;'>Система предиктивной аналитики здоровья на базе ИИ</p>", unsafe_allow_html=True)
st.divider()

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
    
    # --- СОХРАНЕНИЕ В БАЗУ ДАННЫХ ---
    try:
        supabase.table("health_analytics").insert({
            "age": int(in_age),
            "bmi": float(in_bmi),
            "hr_rest": int(in_hr),
            "vo2_max": float(in_vo2),
            "sleep_hours": float(in_sleep),
            "stress_level": int(in_stress),
            "glucose": float(in_glc),
            "bio_age_result": float(prediction)
        }).execute()
    except Exception as e:
        print(f"Ошибка сохранения: {e}") # Юзер этого не увидит

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