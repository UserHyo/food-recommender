import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import requests
from datetime import datetime

@st.cache_resource
def load_preprocessors():
    with open("preprocessors/streamlit_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("preprocessors/streamlit_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    return encoders, scalers

@st.cache_resource
def load_models():
    model_names = ["BrunchSalad", "Noodles", "RiceDishes", "SideDish", "SoupStew", "StirFryGrill"]
    models = {}
    for name in model_names:
        models[name] = joblib.load(f"models/{name}.pkl")
    return models

@st.cache_data
def load_food_data():
    return pd.read_csv("cleaned_food.csv")


# 지역명 → 위도/경도
region_coords = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대전": (36.3504, 127.3845),
    "광주": (35.1595, 126.8526),
    "제주": (33.4996, 126.5312),
    "강릉": (37.7519, 128.8761),
    "대구": (35.8722, 128.6025),
    "수원": (37.2636, 127.0286),
    "청주": (36.6424, 127.4890)
}


# 격자 변환 함수
def dfs_xy_conv(lat, lon):
    import math
    RE = 6371.00877
    GRID = 5.0
    SLAT1, SLAT2 = 30.0, 60.0
    OLON, OLAT = 126.0, 38.0
    XO, YO = 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / \
         math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) /
                  math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = (math.tan(math.pi * 0.25 + slat1 * 0.5) ** sn) * math.cos(slat1) / sn
    ro = re * sf / (math.tan(math.pi * 0.25 + olat * 0.5) ** sn)
    ra = re * sf / (math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5) ** sn)
    theta = lon * DEGRAD - olon
    theta *= sn
    x = ra * math.sin(theta) + XO
    y = ro - ra * math.cos(theta) + YO
    return int(x + 1.5), int(y + 1.5)

API_KEY = "DEaTeAeMY+/ZCys9LTGzBk/MnsJg8VJSGr7h5yrG94i8/FSzVyxUgMsVAM1E3B4XEmYhiTRt5a/fxW+ODvMJ6w=="

def get_weather_data(region_name):
    """
    🔄 실제 기상청 API를 호출하지 않고,
    지역명(region_name)에 따라 임시 날씨 데이터를 반환합니다.
    Hugging Face에서는 SSL 에러로 인해 기상청 API 호출이 불가능하기 때문입니다.
    """
    # 예시: 지역별 간단한 Mock 데이터 (원하는 경우 확장 가능)
    region_coords = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대전": (36.3504, 127.3845),
    "광주": (35.1595, 126.8526),
    "제주": (33.4996, 126.5312),
    "강릉": (37.7519, 128.8761),
    "대구": (35.8722, 128.6025),
    "수원": (37.2636, 127.0286),
    "청주": (36.6424, 127.4890)
}


    # 기본값 (지역이 없을 경우)
    return mock_weather_by_region.get(region_name, {
        "TA_AVG": 16.0,
        "HM_AVG": 60.0,
        "WS_AVG": 1.5,
        "RN_DAY": 0.0
    })

def preprocess_input(gender, age_group, region, weather, encoders, scalers):
    gender_enc = encoders["gender"].transform([gender])[0]
    age_enc = encoders["age_group"].transform([age_group])[0]
    region_enc = encoders["region"].transform([region])[0]

    ta = scalers["TA_AVG"].transform([[weather["TA_AVG"]]])[0][0]
    hm = scalers["HM_AVG"].transform([[weather["HM_AVG"]]])[0][0]
    ws = scalers["WS_AVG"].transform([[weather["WS_AVG"]]])[0][0]
    rn = scalers["RN_DAY"].transform([[weather["RN_DAY"]]])[0][0]

    today = datetime.today()
    return pd.DataFrame([{
        "Gender": gender_enc,
        "Age_Group": age_enc,
        "Region": region_enc,
        "TA_AVG": ta,
        "HM_AVG": hm,
        "WS_AVG": ws,
        "RN_DAY": rn,
        "Month_sin": np.sin(2 * np.pi * today.month / 12),
        "Month_cos": np.cos(2 * np.pi * today.month / 12),
        "Day_sin": np.sin(2 * np.pi * today.day / 31),
        "Day_cos": np.cos(2 * np.pi * today.day / 31),
        "is_weekend": int(today.weekday() >= 5)
    }])

st.title("🍱 날씨 기반 음식 추천 시스템")
st.markdown("#### 오늘 날씨에 가장 인기 있을 음식 카테고리를 추천합니다.")

encoders, scalers = load_preprocessors()
models = load_models()
df_food = load_food_data()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("👤 성별", encoders["gender"].classes_)
    age_group = st.selectbox("🎂 연령대", encoders["age_group"].classes_)
with col2:
    region = st.selectbox("📍 지역", list(region_coords.keys()))

if st.button("🔍 추천받기"):
    weather = get_weather_data(region)
    
    # ✅ 날씨 요약 출력 (metric UI)
    st.markdown("### ☁️ 오늘의 날씨")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("기온", f"{weather['TA_AVG']}°C")
    col2.metric("습도", f"{weather['HM_AVG']}%")
    col3.metric("풍속", f"{weather['WS_AVG']} m/s")
    col4.metric("강수량", f"{weather['RN_DAY']} mm")

    X = preprocess_input(gender, age_group, region, weather, encoders, scalers)

    scores = {}
    for group, model in models.items():
        try:
            scores[group] = model.predict(X)[0]
        except:
            st.error(f"❌ {group} 예측 실패")

    if scores:
        best_group = max(scores, key=scores.get)
        category_map = {
            "BrunchSalad": "브런치/샐러드", "Noodles": "면 요리", "RiceDishes": "밥/죽/덮밥",
            "SideDish": "반찬류", "SoupStew": "찌개/국/탕", "StirFryGrill": "볶음/구이"
        }
        korean_group = category_map[best_group]

        # ✅ 추천 음식 출력
        food_list = df_food[df_food["CKG_GROUP"] == korean_group]["CKG_NM"].dropna().unique()
        if len(food_list) > 0:
            st.markdown("### 🍽️ 오늘의 추천 음식")
            cols = st.columns(len(food_list[:5]))  # 최대 5개
            for col, food in zip(cols, np.random.choice(food_list, size=min(5, len(food_list)), replace=False)):
                col.markdown(
    f"<div style='font-size:16px; font-weight:500; text-align:center;'>🥢{food}</div>",
    unsafe_allow_html=True
)
