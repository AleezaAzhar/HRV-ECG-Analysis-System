import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from src.load_data import load_ecg

# =========================================================
# 🧠 CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("🫀 ICU ECG + FULL HRV ANALYSIS ENGINE")

# =========================================================
# 📂 SIDEBAR
# =========================================================
st.sidebar.header("🫀 ICU LIVE CONTROL")

record_name = st.sidebar.selectbox(
    "Choose ECG record",
    ["16265", "16773", "16420"]
)

fs_override = st.sidebar.number_input("Sampling Frequency Override", value=0)
window_size = st.sidebar.slider("ECG Window Size", 200, 2000, 800)
speed = st.sidebar.slider("Speed", 1, 20, 5)

# =========================================================
# 📥 LOAD DATA
# =========================================================
signal, fs, _ = load_ecg(record_name)

if fs_override > 0:
    fs = fs_override

t = np.arange(len(signal)) / fs

# =========================================================
# 🪟 WINDOW
# =========================================================
duration = 10
samples = int(duration * fs)

sig = signal[:samples]
t_sig = t[:samples]

# =========================================================
# 🔊 NOISE
# =========================================================
np.random.seed(7)
raw_signal = sig + 0.2*np.sin(0.5*np.pi*t_sig) + 0.08*np.random.randn(len(sig))

# =========================================================
# 🧪 FILTER
# =========================================================
def adaptive_bandpass(x, fs):
    low = 0.5
    high = min(40, fs/2 - 1)
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, x)

filtered = adaptive_bandpass(raw_signal, fs)

# =========================================================
# ❤️ R PEAKS
# =========================================================
peaks, _ = find_peaks(filtered, distance=int(0.5*fs), height=np.mean(filtered))
r_win = peaks[peaks < samples]

rr = np.diff(r_win) / fs * 1000 if len(r_win) > 2 else np.array([])

# =========================================================
# 🧠 SAMPLE ENTROPY FUNCTION
# =========================================================
def sample_entropy(signal, m=2, r=None):
    signal = np.array(signal)
    N = len(signal)

    if N < 10:
        return 0

    if r is None:
        r = 0.2 * np.std(signal)

    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N-m+1)])
        C = np.sum([
            np.sum(np.max(np.abs(x - x[i]), axis=1) <= r) - 1
            for i in range(len(x))
        ])
        return C / (N - m + 1)

    phi_m = _phi(m)
    phi_m1 = _phi(m+1)

    if phi_m == 0 or phi_m1 == 0:
        return 0

    return -np.log(phi_m1 / phi_m)

# =========================================================
# 📊 HRV CALCULATIONS
# =========================================================
if len(rr) > 2:
    hr = 60000/np.mean(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    sdnn = np.std(rr)

    diff_rr = np.abs(np.diff(rr))
    nn50 = np.sum(diff_rr > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100
else:
    hr = rmssd = sdnn = nn50 = pnn50 = 0
mean_rr = np.mean(rr) if len(rr) > 2 else 0
# =========================================================
# 🧠 NONLINEAR METRICS
# =========================================================
if len(rr) > 3:
    rr1 = rr[:-1]
    rr2 = rr[1:]
    sd1 = np.sqrt(np.var(rr2 - rr1) / 2)
    sd2 = np.sqrt(2*np.var(rr) - (sd1**2))
else:
    sd1 = sd2 = 0

if len(rr) > 10:
    sampen = sample_entropy(rr)
else:
    sampen = 0
# =========================================================
# 🟢 LIVE ECG
# =========================================================
st.subheader("🫀 Real-Time ECG")

# initialize pointer
if "ptr" not in st.session_state:
    st.session_state.ptr = 0

step = speed
start = st.session_state.ptr
end = start + window_size

# boundary check
if end >= len(signal):
    st.session_state.ptr = 0
    start = 0
    end = window_size

window_signal = signal[start:end]

# update pointer
st.session_state.ptr += step

# 🔥 IMPORTANT: dynamic placeholder
placeholder = st.empty()

fig_live = go.Figure()
fig_live.add_trace(go.Scatter(y=window_signal, mode="lines"))

placeholder.plotly_chart(fig_live, use_container_width=True)
# =========================================================
# 🟢 FILTERED ECG
# =========================================================
st.subheader("🟢 Filtered ECG + R Peaks")

fig = go.Figure()
fig.add_trace(go.Scatter(y=filtered, mode="lines"))

if len(r_win) > 0:
    fig.add_trace(go.Scatter(
        x=r_win,
        y=filtered[r_win],
        mode="markers"
    ))

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 📊 HRV DASHBOARD
# =========================================================
st.subheader("📊 HRV Metrics Overview")

col1, col2, col3 = st.columns(3)

# ================= METRICS =================
with col1:
    st.metric("Heart Rate", f"{hr:.2f}")
    st.metric("Mean RR (ms)", f"{mean_rr:.2f}")   # ✅ ADD THIS
    st.metric("RMSSD", f"{rmssd:.2f}")
    st.metric("SDNN", f"{sdnn:.2f}")
    st.metric("pNN50", f"{pnn50:.2f}")
    st.metric("SD1", f"{sd1:.2f}")
    st.metric("SD2", f"{sd2:.2f}")
    st.metric("SampEn", f"{sampen:.3f}")
# ================= GLOBAL MAP =================
with col2:
    st.markdown("### 🌍 Poincaré Plot (Global)")

    fig_g = go.Figure()

    if len(rr) > 3:
        x = rr[:-1]
        y = rr[1:]

        # Scatter points
        fig_g.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="RR Points"
        ))

        # Identity line (y = x)
        fig_g.add_shape(
            type="line",
            x0=min(rr), y0=min(rr),
            x1=max(rr), y1=max(rr),
            line=dict(dash="dash")
        )

        # Center of ellipse
        mean_val = np.mean(rr)

        # Ellipse (SD1 & SD2)
        theta = np.linspace(0, 2*np.pi, 100)

        ellipse_x = mean_val + sd2 * np.cos(theta)
        ellipse_y = mean_val + sd1 * np.sin(theta)

        fig_g.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode="lines",
            name="SD1-SD2 Ellipse"
        ))

        fig_g.update_layout(
            xaxis_title="RR(n) ms",
            yaxis_title="RR(n+1) ms",
            dragmode="zoom"
        )

    st.plotly_chart(fig_g, use_container_width=True)
    theta = np.linspace(0, 2*np.pi, 100)

mean_val = np.mean(rr)

# rotation (45 degrees)
angle = np.pi / 4

ellipse_x = mean_val + sd2*np.cos(theta)*np.cos(angle) - sd1*np.sin(theta)*np.sin(angle)
ellipse_y = mean_val + sd2*np.cos(theta)*np.sin(angle) + sd1*np.sin(theta)*np.cos(angle)

fig_g.add_trace(go.Scatter(
    x=ellipse_x,
    y=ellipse_y,
    mode="lines",
    name="Rotated Ellipse"
))
# ================= LOCAL MAP =================
with col3:
    st.markdown("### 📍 Local Return Map")

    fig_l = go.Figure()

    if len(rr) > 10:
        rr_local = rr[-15:]
        rr_local = (rr_local - np.mean(rr_local)) / (np.std(rr_local)+1e-6)

        fig_l.add_trace(go.Scatter(
            x=rr_local[:-1],
            y=rr_local[1:],
            mode="markers"
        ))

    st.plotly_chart(fig_l, use_container_width=True)

# =========================================================
# 📡 FREQUENCY ANALYSIS
# =========================================================
st.subheader("📡 Frequency Analysis")

if len(rr) > 5:
    rr_time = np.cumsum(rr)/1000
    rr_time = np.insert(rr_time, 0, 0)
    rr_vals = np.insert(rr, 0, rr[0])

    f_interp = interp1d(rr_time, rr_vals, kind="cubic")
    t_interp = np.linspace(rr_time[0], rr_time[-1], 256)
    rr_interp = f_interp(t_interp)

    f, pxx = welch(rr_interp, fs=4)

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=f, y=pxx, mode="lines"))

    st.plotly_chart(fig_s, use_container_width=True)