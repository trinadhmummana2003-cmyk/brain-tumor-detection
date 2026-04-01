import streamlit as st
from PIL import Image
from predict import predict_tumor
import base64
import os

st.set_page_config(page_title="DeepNeuroVision", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def get_base64_video(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# -----------------------------
# Session State
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result_data" not in st.session_state:
    st.session_state.result_data = None

# -----------------------------
# Video
# -----------------------------
VIDEO_PATH = "assets/brain_bg.mp4"
video_base64 = None
if os.path.exists(VIDEO_PATH):
    video_base64 = get_base64_video(VIDEO_PATH)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: #061522;
    color: white;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Navbar */
.navbar {
    width: 100%;
    background: #ffffff;
    padding: 18px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-sizing: border-box;
    border-bottom: 1px solid #e9eef3;
}

.logo {
    font-size: 22px;
    font-weight: 700;
    color: #2d4059;
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Hero */
.hero {
    position: relative;
    width: 100%;
    height: 88vh;
    overflow: hidden;
    background: #020b14;
}

.hero-video {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 100%;
    height: 100%;
    object-fit: cover;
    transform: translate(-50%, -50%);
    opacity: 0.33;
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.52);
}

.hero-content {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
    width: 85%;
    z-index: 2;
}

.hero-title {
    font-size: 72px;
    font-weight: 800;
    line-height: 1.08;
    color: white;
    margin-bottom: 18px;
}

.hero-subtitle {
    font-size: 20px;
    color: #e5eef7;
    line-height: 1.7;
    max-width: 900px;
    margin: 0 auto 28px auto;
}

.hero-button-wrap {
    max-width: 340px;
    margin: 0 auto;
}

/* Generic page section */
.page-wrap {
    padding: 40px 60px;
    background: linear-gradient(135deg, #071624, #0b2940, #103b56);
    min-height: 100vh;
}

.section-box {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.section-title {
    color: #9fd9ff;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 10px;
}

.section-text {
    color: rgba(255,255,255,0.90);
    font-size: 17px;
    line-height: 1.7;
}

.form-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 30px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.result-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

/* Inputs */
label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
    color: #e8f4ff !important;
    font-weight: 600 !important;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.95) !important;
    color: #111 !important;
    border-radius: 12px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    background: #1e88ff;
    color: white;
    font-size: 17px;
    font-weight: 600;
    height: 50px;
}

.stButton > button:hover {
    background: #1477ea;
    color: white;
}

/* Nav buttons row tweak */
.nav-btn .stButton > button {
    background: transparent !important;
    color: #314861 !important;
    border-radius: 0 !important;
    height: 42px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
    box-shadow: none !important;
}

.nav-btn .stButton > button:hover {
    color: #1e88ff !important;
    border-bottom: 2px solid #1e88ff !important;
}

/* Hero button */
.hero-action .stButton > button {
    border-radius: 999px !important;
    height: 58px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    background: #1e88ff !important;
}

/* Utility */
.center-form {
    max-width: 760px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NAVBAR
# -----------------------------
st.markdown("""
<div class="navbar">
    <div class="logo">🧠 DeepNeuroVision</div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4, nav5, nav6, nav7 = st.columns([6, 1, 1, 1.2, 1, 1, 0.4])

with nav2:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Home", key="nav_home"):
        go_to("home")
    st.markdown('</div>', unsafe_allow_html=True)

with nav3:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("About", key="nav_about"):
        go_to("about")
    st.markdown('</div>', unsafe_allow_html=True)

with nav4:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Detection", key="nav_detection"):
        go_to("detection")
    st.markdown('</div>', unsafe_allow_html=True)

with nav5:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Login", key="nav_login"):
        go_to("login")
    st.markdown('</div>', unsafe_allow_html=True)

with nav6:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Signup", key="nav_signup"):
        go_to("signup")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# HOME PAGE
# -----------------------------
if st.session_state.page == "home":
    if video_base64:
        st.markdown(f"""
        <div class="hero">
            <video class="hero-video" autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-title">Advanced Brain Tumor<br>Detection with AI</div>
                <div class="hero-subtitle">
                    Leveraging deep learning to provide fast and accurate preliminary analysis of
                    MRI scans.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-title">Advanced Brain Tumor<br>Detection with AI</div>
                <div class="hero-subtitle">
                    Leveraging deep learning to provide fast and accurate preliminary analysis of
                    MRI scans.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    a, b, c = st.columns([2.7, 1.3, 2.7])
    with b:
        st.markdown('<div class="hero-action">', unsafe_allow_html=True)
        if st.button("ANALYZE MY SCAN →", key="hero_btn"):
            go_to("detection")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif st.session_state.page == "about":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">About DeepNeuroVision</div>
        <div class="section-text">
            DeepNeuroVision is a medical AI interface designed to analyze MRI brain scans
            using a deep learning model. The system classifies scans into tumor categories
            such as glioma, meningioma, pituitary, or no tumor and presents the result in
            a clean hospital-style interface.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# LOGIN PAGE
# -----------------------------
elif st.session_state.page == "login":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    mid1, mid2, mid3 = st.columns([1.5, 2, 1.5])
    with mid2:
        st.markdown("""
        <div class="form-card">
            <div class="section-title">Login</div>
            <div class="section-text">Sign in to access the medical portal interface.</div>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("Email")
        st.text_input("Password", type="password")
        st.button("Sign In")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# SIGNUP PAGE
# -----------------------------
elif st.session_state.page == "signup":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    mid1, mid2, mid3 = st.columns([1.5, 2, 1.5])
    with mid2:
        st.markdown("""
        <div class="form-card">
            <div class="section-title">Signup</div>
            <div class="section-text">Create an account for the medical AI portal.</div>
        </div>
        """, unsafe_allow_html=True)
        st.text_input("Full Name")
        st.text_input("Email")
        st.text_input("Password", type="password")
        st.button("Create Account")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DETECTION PAGE
# -----------------------------
elif st.session_state.page == "detection":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
        <div class="section-title">MRI Detection Portal</div>
        <div class="section-text">
            Enter patient details and upload an MRI scan for prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="center-form">', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    st.markdown("### 👤 Patient Information")
    name = st.text_input("Full Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=18)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    phone = st.text_input("Phone Number")
    uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="MRI Preview", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        analyze = st.button("🚀 Analyze Scan")
    with b2:
        if st.button("⬅ Back to Home"):
            go_to("home")

    if analyze:
        if name and phone and uploaded_file is not None:
            img.save("temp.jpg")

            with st.spinner("Analyzing MRI image..."):
                tumor_type, percent, severity, doctor, food, lifestyle = predict_tumor("temp.jpg")

            st.session_state.result_data = {
                "name": name,
                "age": age,
                "gender": gender,
                "phone": phone,
                "image_path": "temp.jpg",
                "tumor_type": tumor_type,
                "percent": percent,
                "severity": severity,
                "doctor": doctor,
                "food": food,
                "lifestyle": lifestyle
            }
            go_to("result")
        else:
            st.warning("Please fill all fields and upload an MRI image.")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RESULT PAGE
# -----------------------------
elif st.session_state.page == "result":
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

    data = st.session_state.result_data

    if data is None:
        st.warning("No result found. Please analyze an MRI image first.")
        if st.button("Go to Detection"):
            go_to("detection")
    else:
        st.markdown("""
        <div class="section-box">
            <div class="section-title">MRI Analysis Result</div>
            <div class="section-text">
                Prediction summary and medical suggestions based on the uploaded MRI scan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h3>👤 Patient Summary</h3>
                <p><b>Name:</b> {data['name']}</p>
                <p><b>Age:</b> {data['age']}</p>
                <p><b>Gender:</b> {data['gender']}</p>
                <p><b>Phone:</b> {data['phone']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="result-card">
                <h3>🧾 Prediction Result</h3>
                <p><b>Tumor Type:</b> {data['tumor_type']}</p>
                <p><b>Tumor Percentage:</b> {data['percent']}%</p>
                <p><b>Severity:</b> {data['severity']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if os.path.exists(data["image_path"]):
                result_img = Image.open(data["image_path"])
                st.image(result_img, caption="Uploaded MRI Scan", use_container_width=True)

            st.markdown(f"""
            <div class="result-card">
                <h3>💡 Medical Suggestions</h3>
                <p><b>Doctor Advice:</b> {data['doctor']}</p>
                <p><b>Food Suggestion:</b> {data['food']}</p>
                <p><b>Lifestyle Advice:</b> {data['lifestyle']}</p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            if st.button("⬅ Back to Detection"):
                go_to("detection")
        with r2:
            if st.button("🏠 Back to Home"):
                go_to("home")

    st.markdown('</div>', unsafe_allow_html=True)