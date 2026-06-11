import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

# [UI 설정: 발표용 최적화]
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 0rem; padding-bottom: 0rem;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("⚡ 퀀트 전략 통합 대시보드")

# [사이드바 설정]
mode = st.sidebar.radio("시뮬레이션 모드 선택", 
    ["1. 심화 모델 (Fat Tail & Jump)", 
     "2. 기본 모델 (QRNG vs PRNG)", 
     "3. 통합 비교 분석"])

ticker = st.sidebar.text_input("종목 코드", value="AAPL").strip().upper()
qrng_option = st.sidebar.selectbox("사용할 QRNG 소스", 
    ["양자 난수 세트 1 (기본)", "양자 난수 세트 2 (추가)", "양자 난수 세트 3 (추가)"])

# [공통 데이터 로드 함수]
@st.cache_data
def get_stock_data(ticker):
    return yf.Ticker(ticker).history(period="1y")['Close'].dropna()

close = get_stock_data(ticker)
S0 = float(close.iloc[-1])
ret = np.log(close / close.shift(1)).dropna()
mu = float(np.mean(ret)) * 252
sigma = float(np.std(ret)) * np.sqrt(252)

# [파일 매핑 및 난수 로드]
file_mapping = {"양자 난수 세트 1 (기본)": "qrng_data_1.bin", "양자 난수 세트 2 (추가)": "qrng_data_2.bin", "양자 난수 세트 3 (추가)": "qrng_data_3.bin"}
try:
    with open(file_mapping[qrng_option], "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
except:
    st.error("파일을 찾을 수 없습니다. 깃허브에 .bin 파일을 업로드하세요.")
    st.stop()

STEPS, PATHS = 252, 200
raw_floats = np.resize(raw_data, STEPS * PATHS)
z_q = norm.ppf(np.clip(raw_floats, 1e-7, 1-1e-7))
z_p = np.random.standard_normal(STEPS * PATHS)

# [각 모드별 상세 로직]
if mode == "1. 심화 모델 (Fat Tail & Jump)":
    st.subheader("심화: 팻 테일 & 점프 확산 시뮬레이션")
    # 팻 테일 로직 (t-분포 이용)
    from scipy.stats import t
    z_fat = t.ppf(np.clip(raw_floats, 1e-7, 1-1e-7), df=4) * np.sqrt((4-2)/4)
    res = np.zeros((STEPS, PATHS))
    res[0] = S0
    for t_step in range(1, STEPS):
        z_t = z_fat[(t_step-1)*PATHS : t_step*PATHS]
        res[t_step] = res[t_step-1] * np.exp((mu/252 - 0.5*(sigma/np.sqrt(252))**2) + (sigma/np.sqrt(252)) * z_t)
    
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(res[:, :10], alpha=0.5)
    st.pyplot(fig)

elif mode == "2. 기본 모델 (QRNG vs PRNG)":
    st.subheader("기본: 난수 품질 비교")
    def run_gbm(z):
        res = np.zeros((STEPS, PATHS))
        res[0] = S0
        for t in range(1, STEPS):
            res[t] = res[t-1] * np.exp((mu/252 - 0.5*(sigma/np.sqrt(252))**2) + (sigma/np.sqrt(252)) * z[(t-1)*PATHS : t*PATHS])
        return res
    res_q, res_p = run_gbm(z_q), run_gbm(z_p)
    
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.hist(res_q[-1], bins=50, alpha=0.5, label='QRNG')
    ax.hist(res_p[-1], bins=50, alpha=0.5, label='PRNG')
    ax.legend()
    st.pyplot(fig)

elif mode == "3. 통합 비교 분석":
    st.subheader("비교 분석: 경로, 원뿔, 분포")
    # 경로 및 원뿔 통합 로직 (16:9 비율 유지)
    dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=STEPS)
    def run_gbm(z):
        res = np.zeros((STEPS, PATHS))
        res[0] = S0
        for t in range(1, STEPS):
            res[t] = res[t-1] * np.exp((mu/252 - 0.5*(sigma/np.sqrt(252))**2) + (sigma/np.sqrt(252)) * z[(t-1)*PATHS : t*PATHS])
        return res
    res_q, res_p = run_gbm(z_q), run_gbm(z_p)
    
    fig, ax = plt.subplots(figsize=(12, 6.75))
    q_q, q_p = np.percentile(res_q, [5, 95], axis=1), np.percentile(res_p, [5, 95], axis=1)
    ax.fill_between(dates, q_q[0], q_q[1], color='blue', alpha=0.3, label='QRNG Range')
    ax.fill_between(dates, q_p[0], q_p[1], color='orange', alpha=0.3, label='PRNG Range')
    st.pyplot(fig)