import streamlit as st
import joblib
import numpy as np

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# Load Model
# =========================
model = joblib.load("model.pkl")


# =========================
# Header
# =========================
st.markdown(
    '<div class="main-title">🩺 Breast Cancer Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based Breast Cancer Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the required feature values below and click Predict "
    "to get the model prediction."
)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📌 Model Information")

    st.write("**Model:** Support Vector Classifier (SVC)")
    st.write("**Kernel:** RBF")
    st.write("**C:** 10")
    st.write("**Gamma:** 0.01")
    st.write("**Preprocessing:** StandardScaler")

    st.divider()

    st.caption(
        "This application is a machine learning project "
        "for breast cancer classification."
    )


# =========================
# Mean Features
# =========================
st.markdown(
    '<div class="section-title">📊 Mean Features</div>',
    unsafe_allow_html=True
)

with st.expander("Enter Mean Features", expanded=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input(
            "Radius Mean",
            min_value=0.0,
            format="%.6f"
        )

        texture_mean = st.number_input(
            "Texture Mean",
            min_value=0.0,
            format="%.6f"
        )

        perimeter_mean = st.number_input(
            "Perimeter Mean",
            min_value=0.0,
            format="%.6f"
        )

    with col2:
        area_mean = st.number_input(
            "Area Mean",
            min_value=0.0,
            format="%.6f"
        )

        smoothness_mean = st.number_input(
            "Smoothness Mean",
            min_value=0.0,
            format="%.6f"
        )

        compactness_mean = st.number_input(
            "Compactness Mean",
            min_value=0.0,
            format="%.6f"
        )

    with col3:
        concavity_mean = st.number_input(
            "Concavity Mean",
            min_value=0.0,
            format="%.6f"
        )

        concave_points_mean = st.number_input(
            "Concave Points Mean",
            min_value=0.0,
            format="%.6f"
        )

        symmetry_mean = st.number_input(
            "Symmetry Mean",
            min_value=0.0,
            format="%.6f"
        )


# =========================
# SE Features
# =========================
st.markdown(
    '<div class="section-title">📐 Standard Error Features</div>',
    unsafe_allow_html=True
)

with st.expander("Enter SE Features", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_se = st.number_input(
            "Radius SE",
            min_value=0.0,
            format="%.6f"
        )

        perimeter_se = st.number_input(
            "Perimeter SE",
            min_value=0.0,
            format="%.6f"
        )

    with col2:
        area_se = st.number_input(
            "Area SE",
            min_value=0.0,
            format="%.6f"
        )

        compactness_se = st.number_input(
            "Compactness SE",
            min_value=0.0,
            format="%.6f"
        )

    with col3:
        concavity_se = st.number_input(
            "Concavity SE",
            min_value=0.0,
            format="%.6f"
        )

        concave_points_se = st.number_input(
            "Concave Points SE",
            min_value=0.0,
            format="%.6f"
        )


# =========================
# Worst Features
# =========================
st.markdown(
    '<div class="section-title">🔬 Worst Features</div>',
    unsafe_allow_html=True
)

with st.expander("Enter Worst Features", expanded=False):

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_worst = st.number_input(
            "Radius Worst",
            min_value=0.0,
            format="%.6f"
        )

        texture_worst = st.number_input(
            "Texture Worst",
            min_value=0.0,
            format="%.6f"
        )

        perimeter_worst = st.number_input(
            "Perimeter Worst",
            min_value=0.0,
            format="%.6f"
        )

        area_worst = st.number_input(
            "Area Worst",
            min_value=0.0,
            format="%.6f"
        )

    with col2:
        smoothness_worst = st.number_input(
            "Smoothness Worst",
            min_value=0.0,
            format="%.6f"
        )

        compactness_worst = st.number_input(
            "Compactness Worst",
            min_value=0.0,
            format="%.6f"
        )

        concavity_worst = st.number_input(
            "Concavity Worst",
            min_value=0.0,
            format="%.6f"
        )

    with col3:
        concave_points_worst = st.number_input(
            "Concave Points Worst",
            min_value=0.0,
            format="%.6f"
        )

        symmetry_worst = st.number_input(
            "Symmetry Worst",
            min_value=0.0,
            format="%.6f"
        )

        fractal_dimension_worst = st.number_input(
            "Fractal Dimension Worst",
            min_value=0.0,
            format="%.6f"
        )


# =========================
# Prediction Button
# =========================
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔍 Predict Diagnosis",
        use_container_width=True
    )


# =========================
# Prediction
# =========================
if predict_button:

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        radius_se,
        perimeter_se,
        area_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    prediction = model.predict(input_data)

    st.divider()

    st.subheader("📋 Prediction Result")

    if prediction[0] == 1:
        st.error(
            "Prediction: Malignant"
        )
    else:
        st.success(
            "Prediction: Benign"
        )