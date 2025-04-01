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
    "강릉": (37.7519, 128.8761)
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
    lat, lon = region_coords[region_name]
    x, y = dfs_xy_conv(lat, lon)
    base_date = datetime.today().strftime('%Y%m%d')
    base_time = "0500"

    url = (
        f"https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        f"?serviceKey={API_KEY}&pageNo=1&numOfRows=1000&dataType=JSON"
        f"&base_date={base_date}&base_time={base_time}&nx={x}&ny={y}"
    )

    res = requests.get(url)
    items = res.json().get("response", {}).get("body", {}).get("items", {}).get("item", [])

    result = {"TA_AVG": None, "HM_AVG": None, "WS_AVG": None, "RN_DAY": 0.0}
    for item in items:
        cat = item["category"]
        val = item["fcstValue"]
        if cat == "TMP":
            result["TA_AVG"] = float(val)
        elif cat == "REH":
            result["HM_AVG"] = float(val)
        elif cat == "WSD":
            result["WS_AVG"] = float(val)
        elif cat == "PCP":
            result["RN_DAY"] = 0.0 if val == "강수없음" else float(val.replace("mm", "").strip())
    return result

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
    st.info(f"📡 '{region}' 지역 날씨: {weather}")
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
        st.success(f"🍲 오늘은 '{korean_group}'이 가장 인기 있을 것 같아요!")

        # 음식 추천
        food_list = df_food[df_food["CKG_GROUP"] == korean_group]["CKG_NM"].dropna().unique()
        if len(food_list) > 0:
            st.markdown("#### 🥢 추천 음식:")
            for food in np.random.choice(food_list, size=min(5, len(food_list)), replace=False):
                st.markdown(f"- {food}")

        st.bar_chart(pd.Series(scores).sort_values(ascending=False))

