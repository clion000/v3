import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import t
import warnings
warnings.filterwarnings('ignore')

# [UI 최적화 설정]
st.set_page_config(page_title="Advanced Quant Simulator", layout="wide", initial_sidebar_state="expanded")
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("⚡ 퀀트 심화 시뮬레이터 (Mean Reversion + Fat Tail + Jump)")

# [통화 포맷팅 헬퍼 함수]
def format_price(value, currency):
    if currency == "₩":
        return f"{currency}{int(value):,}"
    else:
        return f"{currency}{value:,.2f}"

# ==========================================
# 1. 사이드바: 기본 설정 및 파라미터 조절
# ==========================================
st.sidebar.header("📊 분석 기본 설정")
TICKER = st.sidebar.text_input("종목 코드 (예: AAPL, 005930.KS)", value="AAPL").strip().upper()
currency_symbol = "₩" if TICKER.endswith(".KS") or TICKER.endswith(".KQ") else "$"

# [변경됨] 파일 업로드 대신 미리 정의된 파일 목록에서 선택
st.sidebar.markdown("---")
st.sidebar.header("📂 양자 난수 소스 선택")
file_mapping = {
    "양자 난수 세트 1 (기본)": "qrng_data_1.bin",
    "양자 난수 세트 2 (추가)": "qrng_data_2.bin",
    "양자 난수 세트 3 (추가)": "qrng_data_3.bin"
}
qrng_option = st.sidebar.selectbox("사용할 QRNG 데이터", list(file_mapping.keys()))
target_file = file_mapping[qrng_option]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 심화 모델 파라미터 튜닝")
kappa = st.sidebar.slider("평균 회귀 강도 (kappa)", 0.0, 1.0, 0.1, 0.05, help="장기 평균으로 돌아가려는 힘")
df = st.sidebar.slider("팻 테일 자유도 (df)", 3, 30, 8, 1, help="낮을수록 극단적인 폭락/폭등 빈도 증가")
lambda_jump = st.sidebar.slider("연간 점프 빈도 (lambda)", 0.0, 5.0, 0.5, 0.1, help="1년에 발생하는 평균 블랙스완 횟수")
mu_j = st.sidebar.number_input("평균 점프 크기 (mu_j)", -0.50, 0.50, -0.02, 0.01)
sigma_j = st.sidebar.number_input("점프 변동성 (sigma_j)", 0.0, 0.50, 0.10, 0.01)

# ==========================================
# 2. 데이터 로드
# ==========================================
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="1y", progress=False)
    if data.empty:
        return None
    return data

data = load_data(TICKER)

if data is None:
    st.error(f"❌ '{TICKER}' 종목의 데이터를 불러오지 못했습니다. 종목 코드를 확인해주세요.")
    st.stop()

returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
if isinstance(returns, pd.DataFrame):
    returns = returns.iloc[:, 0]

sigma_annual = float(np.std(returns) * np.sqrt(252))
S0 = float(data['Close'].dropna().iloc[-1])
last_date = data['Close'].dropna().index[-1]
theta = np.mean(np.log(data['Close'].dropna())) 

st.markdown(f"**기준일:** {last_date.strftime('%Y-%m-%d')} | **현재가:** {format_price(S0, currency_symbol)} | **연간 변동성:** {sigma_annual*100:.2f}%")

# ==========================================
# 3. QRNG 파일 로드 및 난수 변환
# ==========================================
try:
    with open(target_file, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
except FileNotFoundError:
    st.error(f"❌ '{target_file}' 파일을 찾을 수 없습니다. 깃허브 최상단 경로에 파일이 업로드되어 있는지 확인해주세요.")
    st.stop()

u_data = np.clip(raw_data.astype(np.float32) / 255.0, 1e-7, 1 - 1e-7)

STEPS = 252
NUM_PATHS = 1000
dt = 1 / 252
required_z = STEPS * NUM_PATHS

if len(u_data) < required_z:
    u_data = np.resize(u_data, required_z)

z_fat = t.ppf(u_data[:required_z], df) * np.sqrt((df-2)/df)

# ==========================================
# 4. 시뮬레이션 실행 엔진
# ==========================================
X = np.zeros((STEPS, NUM_PATHS))
X[0] = np.log(S0)
np.random.seed(42)

with st.spinner("시뮬레이션 연산 중..."):
    for i in range(1, STEPS):
        z_t = z_fat[(i-1)*NUM_PATHS : i*NUM_PATHS]
        N = np.random.poisson(lambda_jump * dt, NUM_PATHS)
        J = np.random.normal(mu_j, sigma_j, NUM_PATHS)
        
        drift = kappa * (theta - X[i-1]) * dt
        diffusion = sigma_annual * np.sqrt(dt) * z_t
        jump_effect = N * J
        
        X[i] = X[i-1] + drift + diffusion + jump_effect

    results = np.exp(X)

# ==========================================
# 5. 결과 요약 통계 출력
# ==========================================
final_prices = results[-1, :]
st.markdown("### 📊 1년 후 주가 예측 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("평균 기대 주가", format_price(np.mean(final_prices), currency_symbol))
col2.metric("중앙값 (Median)", format_price(np.median(final_prices), currency_symbol))
col3.metric("최고 예상 주가 (Max)", format_price(np.max(final_prices), currency_symbol))
col4.metric("최저 예상 주가 (Min)", format_price(np.min(final_prices), currency_symbol))
st.markdown("---")

# ==========================================
# 6. 데이터 시각화 (16:9 비율 유지)
# ==========================================
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=STEPS)
quantiles = np.percentile(results, [5, 25, 50, 75, 95], axis=1)

st.subheader("📈 확률 원뿔 (Probability Cone)")
fig1, ax1 = plt.subplots(figsize=(12, 6.75))
ax1.fill_between(future_dates, quantiles[0], quantiles[4], color='blue', alpha=0.1, label='5% - 95% Range')
ax1.fill_between(future_dates, quantiles[1], quantiles[3], color='blue', alpha=0.3, label='25% - 75% Range')
ax1.plot(future_dates, quantiles[2], color='navy', linewidth=2, label='Median Path (50%)')
ax1.axhline(np.exp(theta), color='red', linestyle='--', linewidth=1.5, label='Long-term Mean (Theta)')
ax1.set_ylabel(f"Price ({currency_symbol})")
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
st.pyplot(fig1)

st.subheader("📉 최종 주가 분포 비교 (Fat Tail & Jump)")
fig2, ax2 = plt.subplots(figsize=(12, 6.75))
ax2.hist(final_prices, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
ax2.axvline(S0, color='red', linestyle='dashed', linewidth=2, label=f"Current: {format_price(S0, currency_symbol)}")
ax2.axvline(np.mean(final_prices), color='green', linestyle='dashed', linewidth=2, label="Average")
ax2.axvline(np.exp(theta), color='purple', linestyle='dashed', linewidth=2, label="Target(Theta)")
ax2.set_xlabel(f"Final Price ({currency_symbol})")
ax2.set_ylabel("Frequency")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)
st.pyplot(fig2)