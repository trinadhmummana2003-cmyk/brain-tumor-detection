import streamlit as st
from PIL import Image
from predict import predict_tumor
import base64
import os
import tempfile
import re

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

st.set_page_config(page_title="DeepNeuroVision", layout="wide")

# ============================================================
# Helpers
# ============================================================
def get_base64_video(video_path: str):
    if os.path.exists(video_path):
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def go_to(page_name: str):
    st.session_state.page = page_name
    st.rerun()


def severity_badge_class(severity: str):
    s = str(severity).strip().lower()
    if "high" in s:
        return "sev-high"
    if "moderate" in s or "medium" in s:
        return "sev-medium"
    if "low" in s or "mild" in s or "simple" in s:
        return "sev-low"
    return "sev-ok"


def is_valid_phone(phone: str):
    pattern = r"^[6-9]\d{9}$"
    return re.match(pattern, phone)


def is_fake_number(phone: str):
    if len(set(phone)) == 1:
        return True
    fake_list = ["1234567890", "0123456789", "9876543210"]
    return phone in fake_list


def generate_pdf(data, file_path):
    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#123150"),
        spaceAfter=12
    )

    sub_title_style = ParagraphStyle(
        "SubTitleStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#5f7892"),
        spaceAfter=14
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0b74ff"),
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#17324f")
    )

    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#61798f")
    )

    elements = []

    elements.append(Paragraph("DeepNeuroVision", title_style))
    elements.append(Paragraph("Brain Tumor Detection Report", sub_title_style))

    elements.append(Paragraph("Patient Information", section_style))

    patient_data = [
        ["Patient Name", str(data["name"])],
        ["Age", str(data["age"])],
        ["Gender", str(data["gender"])],
        ["Phone Number", str(data["phone"])]
    ]

    patient_table = Table(patient_data, colWidths=[140, 330])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#17324f")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#d9e7f3")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef6ff")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEADING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Prediction Summary", section_style))

    summary_data = [
        ["Tumor Status", str(data["tumor_status"])],
        ["Tumor Type", str(data["tumor_type"])],
        ["Model Confidence", f"{data['confidence']}%"],
        ["Tumor Percentage", f"{data['percent']}%"],
        ["Severity", str(data["severity"])]
    ]

    summary_table = Table(summary_data, colWidths=[140, 330])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#17324f")),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#d9e7f3")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef6ff")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEADING", (0, 0), (-1, -1), 14),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("Medical Guidance", section_style))
    elements.append(Paragraph("<b>Doctor Advice:</b>", normal_style))
    elements.append(Paragraph(str(data["doctor"]), small_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Food Suggestion:</b>", normal_style))
    elements.append(Paragraph(str(data["food"]), small_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Lifestyle Advice:</b>", normal_style))
    elements.append(Paragraph(str(data["lifestyle"]), small_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Age-based Care Note:</b>", normal_style))
    elements.append(Paragraph(str(data["age_note"]), small_style))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Gender-based Care Note:</b>", normal_style))
    elements.append(Paragraph(str(data["gender_note"]), small_style))
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Clinical Disclaimer", section_style))
    elements.append(
        Paragraph(
            "This report is generated for educational and project demonstration purposes only. "
            "It should not be treated as a final medical diagnosis. "
            "Please consult a qualified doctor or radiologist for clinical confirmation.",
            small_style
        )
    )

    doc.build(elements)


# ============================================================
# Session State
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "result_data" not in st.session_state:
    st.session_state.result_data = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False


# ============================================================
# Assets
# ============================================================
VIDEO_PATH = "assets/brain_bg.mp4"
video_base64 = get_base64_video(VIDEO_PATH)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
#MainMenu, header, footer {visibility: hidden;}

html, body, [class*="css"] {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(24,119,242,0.08), transparent 26%),
        linear-gradient(180deg, #eef5fb 0%, #f6f9fc 45%, #eef4f8 100%);
    color: #11314d;
}

.block-container {
    max-width: 1220px !important;
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.8rem !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 1rem !important;
    align-items: flex-start !important;
}

/* Top bar */
.topbar {
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(210,226,240,0.95);
    border-radius: 24px;
    padding: 16px 22px;
    margin: 14px 0 12px 0;
    box-shadow: 0 14px 36px rgba(24, 40, 72, 0.06);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0b74ff 0%, #26b3ff 100%);
    color: white;
    font-size: 24px;
    box-shadow: 0 12px 26px rgba(11, 116, 255, 0.24);
}

.brand-title {
    font-size: 25px;
    font-weight: 900;
    color: #123150;
    line-height: 1.05;
}

.brand-subtitle {
    font-size: 13px;
    color: #6b859d;
    margin-top: 2px;
}

/* Navigation */
.nav-btn .stButton > button {
    background: rgba(255,255,255,0.94) !important;
    color: #1e496f !important;
    border: 1px solid #d9e8f3 !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 20px rgba(22, 47, 83, 0.05) !important;
    height: 46px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

.nav-btn .stButton > button:hover {
    border-color: #a9d2f0 !important;
    color: #0b74ff !important;
}

/* Hero */
.hero {
    position: relative;
    min-height: 465px;
    border-radius: 34px;
    overflow: hidden;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, #071622 0%, #0d2740 55%, #143a59 100%);
    box-shadow: 0 22px 50px rgba(11, 28, 48, 0.22);
}

.hero-video {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0.34;
}

.hero-overlay {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(115deg, rgba(4,12,20,0.82) 10%, rgba(5,16,27,0.56) 55%, rgba(8,24,39,0.78) 100%),
        radial-gradient(circle at top left, rgba(77,171,247,0.18), transparent 35%);
}

.hero-content {
    position: relative;
    z-index: 2;
    min-height: 465px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 50px 26px;
}

.hero-inner {
    max-width: 760px;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px 15px;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.12);
    color: #c8e8ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.04em;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 62px;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.03em;
}

.hero-subtitle {
    font-size: 23px;
    font-weight: 700;
    color: #a8d7ff;
    margin-bottom: 26px;
}

.hero-btn-wrap {
    max-width: 240px;
    margin: 0 auto;
}

.hero-btn-wrap .stButton > button {
    height: 56px !important;
    border-radius: 999px !important;
    font-size: 17px !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #0b74ff, #28b0ff) !important;
    color: white !important;
    box-shadow: 0 18px 30px rgba(11, 116, 255, 0.28) !important;
}

/* Section headings */
.section-title {
    font-size: 31px;
    font-weight: 900;
    color: #123150;
    margin-top: 10px;
    margin-bottom: 8px;
}

.section-desc {
    font-size: 15px;
    line-height: 1.85;
    color: #6c8296;
    margin-bottom: 18px;
    max-width: 880px;
}

/* Cards */
.glass-card,
.white-card,
.metric-card,
.info-box,
.result-banner,
.highlight-card {
    border-radius: 24px;
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(222,234,244,0.95);
    box-shadow: 0 16px 34px rgba(18, 36, 61, 0.06);
}

.white-card,
.metric-card,
.info-box,
.highlight-card,
.glass-card {
    padding: 22px;
}

.result-banner {
    padding: 24px;
    margin-bottom: 18px;
}

.highlight-card {
    height: 100%;
    position: relative;
    overflow: hidden;
}

.highlight-card:before {
    content: "";
    position: absolute;
    inset: 0 auto auto 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #0b74ff, #33c2ff);
}

.card-title {
    font-size: 20px;
    font-weight: 850;
    color: #123150;
    margin-bottom: 8px;
}

.card-subtitle,
.card-text {
    font-size: 14px;
    line-height: 1.85;
    color: #6d8397;
}

.metric-label {
    font-size: 13px;
    color: #6d8397;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 24px;
    font-weight: 900;
    color: #123150;
}

.result-title {
    font-size: 32px;
    font-weight: 900;
    color: #123150;
    margin-bottom: 6px;
}

.result-subtitle {
    font-size: 15px;
    color: #6e8498;
}

/* Widget inputs */
label, .stTextInput label, .stNumberInput label, .stSelectbox label, .stFileUploader label {
    color: #244b72 !important;
    font-weight: 700 !important;
}

.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.96) !important;
    color: #123150 !important;
    border: 1px solid #d8e7f3 !important;
    border-radius: 16px !important;
    min-height: 50px !important;
    box-shadow: inset 0 1px 2px rgba(13, 35, 58, 0.03);
}

.stFileUploader section {
    background: rgba(255,255,255,0.96) !important;
    border: 1px dashed #bdd8eb !important;
    border-radius: 20px !important;
    padding: 14px !important;
}

.stButton > button {
    width: 100%;
    border: none !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #0b74ff, #2ea8ff) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    height: 50px !important;
    box-shadow: 0 12px 24px rgba(11, 116, 255, 0.18) !important;
}

.stButton > button:hover {
    filter: brightness(1.03);
    color: white !important;
}

.secondary-btn .stButton > button {
    background: rgba(255,255,255,0.96) !important;
    color: #1d466f !important;
    border: 1px solid #d8e7f3 !important;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04) !important;
}

/* Badges and info */
.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 120px;
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 800;
}

.sev-high { background: #ffe7e7; color: #c62828; border: 1px solid #ffcaca; }
.sev-medium { background: #fff3de; color: #b7791f; border: 1px solid #ffe1a2; }
.sev-low { background: #eafbf1; color: #1f8c55; border: 1px solid #c7eed8; }
.sev-ok { background: #ebf4ff; color: #156cc2; border: 1px solid #d2e6ff; }

.info-item {
    padding: 12px 0;
    border-bottom: 1px solid #edf3f8;
}

.info-item:last-child {
    border-bottom: none;
}

.info-key {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #73a5d0;
    font-weight: 800;
    margin-bottom: 4px;
}

.info-value {
    font-size: 16px;
    line-height: 1.8;
    color: #17324f;
}

.disclaimer {
    background: linear-gradient(180deg, #f8fbff 0%, #f4f9fe 100%);
    border-left: 4px solid #38a6ff;
    border-radius: 18px;
    padding: 16px 18px;
    color: #61798f;
    line-height: 1.8;
    font-size: 14px;
}

.footer-note {
    text-align: center;
    font-size: 13px;
    color: #7a90a5;
    padding: 8px 0 0 0;
}

/* Streamlit metric styling */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(222,234,244,0.95);
    border-radius: 20px;
    padding: 14px 18px;
    box-shadow: 0 12px 26px rgba(18, 36, 61, 0.05);
}

/* Mobile */
@media (max-width: 980px) {
    .hero-title { font-size: 40px; }
    .hero-subtitle { font-size: 18px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Top Bar
# ============================================================
st.markdown(
    """
<div class="topbar">
    <div class="brand-row">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-title">DeepNeuroVision</div>
            <div class="brand-subtitle">Brain tumor detection with CNN-based MRI analysis</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

_, nav_b, nav_c, nav_d, nav_e = st.columns([6, 1, 1, 1.2, 1.2])

with nav_b:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Home", key="nav_home"):
        go_to("home")
    st.markdown("</div>", unsafe_allow_html=True)

with nav_c:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("About", key="nav_about"):
        go_to("about")
    st.markdown("</div>", unsafe_allow_html=True)

with nav_d:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Detection", key="nav_detection"):
        go_to("detection")
    st.markdown("</div>", unsafe_allow_html=True)

with nav_e:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("Results", key="nav_result"):
        go_to("result")
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Home Page
# ============================================================
if st.session_state.page == "home":
    if video_base64:
        st.markdown(
            f"""
        <div class="hero">
            <video class="hero-video" autoplay muted loop playsinline>
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            </video>
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-inner">
                    <div class="hero-tag">Clinical-style MRI screening interface</div>
                    <div class="hero-title">DeepNeuroVision</div>
                    <div class="hero-subtitle">Brain Tumor Detection</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="hero">
            <div class="hero-overlay"></div>
            <div class="hero-content">
                <div class="hero-inner">
                    <div class="hero-tag">Clinical-style MRI screening interface</div>
                    <div class="hero-title">DeepNeuroVision</div>
                    <div class="hero-subtitle">Brain Tumor Detection</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    cta1, cta2, cta3 = st.columns([1.3, 1, 1.3])
    with cta2:
        st.markdown('<div class="hero-btn-wrap">', unsafe_allow_html=True)
        if st.button("Start Detection", key="hero_cta"):
            go_to("detection")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-title">Detection Workflow Highlights</div>
        <div class="section-desc">
            Designed with a realistic medical analysis feel using premium clinical colors, cleaner hierarchy, soft glass-style cards, and a focused workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="card-title">Brain Tumor Detection</div>
                <div class="card-text">
                    Uses the trained CNN model to analyze the uploaded brain MRI scan and generate a structured tumor prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="card-title">Patient Intake + MRI Upload</div>
                <div class="card-text">
                    Collects patient information and MRI scan image input in a simple hospital-style workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h3:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="card-title">Professional Results</div>
                <div class="card-text">
                    Presents tumor status, tumor type, confidence, severity, patient summary, and guidance in clean premium cards.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer" style="margin-top:16px;">
            <b>Clinical Note:</b> This project provides a CNN model-based preliminary prediction for educational demonstration only.
            Final diagnosis must be verified by a qualified doctor or radiologist.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# About Page
# ============================================================
elif st.session_state.page == "about":
    st.markdown(
        """
        <div class="section-title">About DeepNeuroVision</div>
        <div class="section-desc">
            DeepNeuroVision is a brain tumor detection interface built for academic demonstration.
            It combines patient intake, MRI upload, CNN model prediction, and a structured result dashboard with a realistic medical presentation style.
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)

    with a1:
        st.markdown(
            """
            <div class="white-card">
                <div class="card-title">Core Features</div>
                <div class="card-text">
                    MRI upload, patient intake, CNN model prediction, and organized result presentation in a professional clinical layout.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
            <div class="white-card">
                <div class="card-title">Project Purpose</div>
                <div class="card-text">
                    Designed to provide a polished and review-ready interface for demonstrating brain tumor detection using MRI scans.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Detection Page
# ============================================================
elif st.session_state.page == "detection":
    st.markdown(
        """
        <div class="section-title">Patient Intake and MRI Upload</div>
        <div class="section-desc">
            Fill in patient details, upload the MRI image, verify the preview, and run the CNN model analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.02, 0.98])

    with left_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">Patient Information</div>
                <div class="card-subtitle">Enter the patient details carefully before running the scan analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=18)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        phone = st.text_input("Phone Number")

    with right_col:
        st.markdown(
            """
            <div class="glass-card">
                <div class="card-title">MRI Scan Upload</div>
                <div class="card-subtitle">Upload a brain MRI image in JPG, JPEG, or PNG format.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            preview_img = Image.open(uploaded_file).convert("RGB")
            st.image(preview_img, caption="Uploaded MRI Preview", use_container_width=True)
        else:
            st.markdown(
                """
                <div class="white-card">
                    <div class="card-text">No MRI image uploaded yet. The image preview will appear here after upload.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    b1, b2, b3 = st.columns(3)

    with b1:
        analyze = st.button("Analyze MRI Scan", key="analyze_btn")

    with b2:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("Clear Result", key="clear_result"):
            st.session_state.result_data = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
        if st.button("Back to Home", key="back_home"):
            go_to("home")
        st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        if not name or not phone or uploaded_file is None:
            st.warning("Please fill all patient details and upload an MRI image before analysis.")
        elif not is_valid_phone(phone):
            st.error("Enter a valid 10-digit Indian phone number starting with 6, 7, 8, or 9.")
        elif is_fake_number(phone):
            st.error("Invalid phone number detected. Please enter a real phone number.")
        else:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_path = temp_file.name

                img.save(temp_path)

                with st.spinner("Running CNN model analysis on MRI image..."):
                    result = predict_tumor(temp_path, age=age, gender=gender)

                st.session_state.result_data = {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "phone": phone,
                    "image_path": temp_path,
                    "tumor_detected": result["tumor_detected"],
                    "tumor_status": result["tumor_status"],
                    "tumor_type": result["tumor_type"],
                    "confidence": result["confidence"],
                    "percent": result["tumor_percentage"],
                    "severity": result["severity"],
                    "doctor": result["doctor"],
                    "food": result["food"],
                    "lifestyle": result["lifestyle"],
                    "age_note": result["age_note"],
                    "gender_note": result["gender_note"],
                }

                st.session_state.pdf_ready = False
                go_to("result")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ============================================================
# Result Page
# ============================================================
elif st.session_state.page == "result":
    data = st.session_state.result_data

    if data is None:
        st.markdown(
            """
            <div class="white-card">
                <div class="card-title">No analysis result available</div>
                <div class="card-subtitle">Please upload an MRI image and run the detection process first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        with c2:
            if st.button("Go to Detection", key="go_detection_empty"):
                go_to("detection")

    else:
        badge_class = severity_badge_class(data["severity"])

        head1, head2 = st.columns([5, 1.2])

        with head1:
            st.markdown(
                """
                <div class="result-banner">
                    <div class="result-title">MRI Analysis Completed</div>
                    <div class="result-subtitle">CNN model prediction summary for the uploaded brain MRI scan.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with head2:
            st.markdown(
                f"""
                <div style="padding-top:18px;">
                    <div class="badge {badge_class}">{data['severity']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tumor Status</div>
                <div class="metric-value">{data['tumor_status']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tumor Type</div>
                <div class="metric-value">{data['tumor_type']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{data['confidence']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tumor Percentage</div>
                <div class="metric-value">{data['percent']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Severity</div>
                <div class="metric-value">{data['severity']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        rc1, rc2 = st.columns([0.96, 1.04])

        with rc1:
            st.markdown(
                f"""
                <div class="info-box">
                    <div class="card-title">Patient Summary</div>
                    <div class="info-item"><div class="info-key">Full Name</div><div class="info-value">{data['name']}</div></div>
                    <div class="info-item"><div class="info-key">Age</div><div class="info-value">{data['age']}</div></div>
                    <div class="info-item"><div class="info-key">Gender</div><div class="info-value">{data['gender']}</div></div>
                    <div class="info-item"><div class="info-key">Phone Number</div><div class="info-value">{data['phone']}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if os.path.exists(data["image_path"]):
                st.markdown(
                    """
                    <div class="info-box">
                        <div class="card-title">Uploaded MRI Scan</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                result_img = Image.open(data["image_path"])
                st.image(result_img, use_container_width=True)

        with rc2:
            st.markdown(
                f"""
                <div class="info-box">
                    <div class="card-title">Medical Guidance Panel</div>
                    <div class="card-subtitle">Supportive information generated from the predicted output.</div>
                    <div class="info-item"><div class="info-key">Doctor Advice</div><div class="info-value">{data['doctor']}</div></div>
                    <div class="info-item"><div class="info-key">Food Suggestion</div><div class="info-value">{data['food']}</div></div>
                    <div class="info-item"><div class="info-key">Lifestyle Advice</div><div class="info-value">{data['lifestyle']}</div></div>
                    <div class="info-item"><div class="info-key">Age-based Care Note</div><div class="info-value">{data['age_note']}</div></div>
                    <div class="info-item"><div class="info-key">Gender-based Care Note</div><div class="info-value">{data['gender_note']}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="disclaimer">
                    <b>Clinical Disclaimer:</b> This output is a CNN model-based educational prediction and should be used only as a preliminary reference.
                    A radiologist or qualified doctor must verify all findings before any medical decision is made.
                </div>
                """,
                unsafe_allow_html=True,
            )

        pdf_col1, pdf_col2 = st.columns([1, 2])

        with pdf_col1:
            if st.button("Generate PDF Report", key="generate_pdf_btn"):
                pdf_path = "Brain_Tumor_Report.pdf"
                generate_pdf(data, pdf_path)
                st.session_state.pdf_ready = True

        with pdf_col2:
            if st.session_state.pdf_ready and os.path.exists("Brain_Tumor_Report.pdf"):
                with open("Brain_Tumor_Report.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_file,
                        file_name="Brain_Tumor_Report.pdf",
                        mime="application/pdf",
                        key="download_pdf_btn"
                    )

        d1, d2, d3 = st.columns(3)

        with d1:
            if st.button("Analyze Another Scan", key="again"):
                go_to("detection")

        with d2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Back to Home", key="home_from_result"):
                go_to("home")
            st.markdown("</div>", unsafe_allow_html=True)

        with d3:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            if st.button("Refresh Result", key="refresh_result"):
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-note">
            DeepNeuroVision • Professional academic interface for brain tumor detection using a CNN model
        </div>
        """,
        unsafe_allow_html=True,
    )