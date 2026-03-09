import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import date

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="wide"
)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        # Note: If the warning persists, re-run your training script 
        # to generate these files using your current Python/XGBoost version.
        model = joblib.load('flight_model_v2.pkl')
        details = joblib.load('model_details_v2.pkl')
        return model, details
    except:
        return None, None

model, details = load_assets()


# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f0f2f6; }
    .prediction-card {
        background: #1E3A8A; 
        padding: 30px; 
        border-radius: 15px; 
        text-align: center; 
        color: white;
        box-shadow: 0 10px 25px rgba(30, 58, 138, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("✈️ Real-time Flight Fare Prediction")
st.markdown("Fill in the details to get an estimated price.")

if model is None:
    st.error("Optimized Model assets not found. Please run the training script first.")
    st.stop()

# --- INPUT SECTION ---
with st.container():
    row1_col1, row1_col2,row1_col3 = st.columns(3)
    with row1_col1:
        airline = st.selectbox("Airline", options=sorted(details['airlines']))
    with row1_col2:
        Boarding = st.selectbox("Boarding", options=sorted(details['cities']), index=0)
    with row1_col3:
        # FIXED: Variable name set to 'Destination'
        Destination = st.selectbox("Destination", options=sorted(details['cities']), index=1)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        Class = st.radio("Cabin Class", ["Economy", "Business"], horizontal=True)
    with row2_col2:
        Date_to_travel = st.date_input("Date of Journey", min_value=date.today())
    with row2_col3:
        stops = st.selectbox("Stops", options=["Non-stop", "1-stop", "2+-stop"])

# --- PROCESSING & PREDICTION ---
days_left = (Date_to_travel - date.today()).days

# 1. Initialize a DataFrame with zeros using the EXACT columns the model was trained on
input_df = pd.DataFrame(0, index=[0], columns=details['columns'])

# 2. Map Numerical/Direct Fields
if 'days_left' in input_df.columns:
    input_df['days_left'] = days_left

# 3. Map Categorical Fields (One-Hot Encoding matching)
# Airline
airline_col = f'airline_{airline}'
if airline_col in input_df.columns:
    input_df[airline_col] = 1

# Source
source_col = f'source_city_{Boarding}'
if source_col in input_df.columns:
    input_df[source_col] = 1

# Destination
dest_col = f'destination_city_{Destination}'
if dest_col in input_df.columns:
    input_df[dest_col] = 1

# Class (Your model has 'class_Economy', let's set it based on selection)
if 'class_Economy' in input_df.columns:
    input_df['class_Economy'] = 1 if Class == "Economy" else 0
if 'class_Business' in input_df.columns:
    input_df['class_Business'] = 1 if Class == "Business" else 0

# 4. Map Stops (Mapping your selectbox to the model's specific stop columns)
if stops == "1-stop" and 'stops_one' in input_df.columns:
    input_df['stops_one'] = 1
elif stops == "2+-stop" and 'stops_two_or_more' in input_df.columns:
    input_df['stops_two_or_more'] = 1
# (Non-stop is usually the 'baseline' where both are 0)

# 5. Ensure Column Order (Crucial for XGBoost)
input_df = input_df[details['columns']]

# --- RESULTS & INNOVATION LAYOUT ---
st.markdown("<br>", unsafe_allow_html=True)
left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
prediction = model.predict(input_df)[0]

with center_col:
    st.markdown(f"""
    <div style="background: #1E3A8A; padding: 25px; border-radius: 10px; text-align: center; color: white;">
                <p style="text-transform: uppercase; font-size: 0.8rem; letter-spacing: 1.5px; opacity: 0.8; margin-bottom: 5px;">Predicted fare</p>
                <h1 style="margin: 0; font-size: 3rem; color: #FFFFFF;">₹ {round(float(prediction),2 ):,}</h1>
                <p style="margin-top: 10px; font-size: 0.9rem; color: #93C5FD;">
            </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Booking Advice")
    avg_fare = details['avg_price']
    
    if prediction < avg_fare * 0.92:
        st.success("🎯 **BUY NOW!** This fare is statistically rare for this route. Save in more.")
    elif prediction > avg_fare * 1.15:
        st.error("🚨 **WAIT!** It's a High price. Wait for a 10-15% dip.")
    else:
        st.info("⚖️ **NEUTRAL.** This is the standard market fare. Book if you want to travel.")
    
    #st.progress(min(max((prediction / (avg_fare * 2)), 0.1), 1.0))
   
# --- UNIQUE FEATURE: EXPLAINABLE AI ---
st.write("---")
st.subheader("🔍  See Your Decision")

try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], 
                    matplotlib=True, show=False, plot_cmap="PkYlGn")
    st.pyplot(plt.gcf())
    st.caption("Red features increase price | Blue features decrease price")
except Exception as e:
    st.info("Explainer visualization is loading...")

# --- TRAVEL ADD-ONS ---
st.write("---")
st.subheader(f"🛠️ Travel Essentials: {Boarding} to {Destination}")
tab1, tab2, tab3, tab4 = st.tabs(["🏨 Stay & Dine", "🚕 Taxi", " 📍🏰 Tourist Sightings", "🎫 Deals & Coupons"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 🛌 Top Stays")
        st.info(f"Checking availability in {Destination}...")
    with c2:
        st.markdown("##### 🍴 Local Eats")
        st.success("Top Rated Fine Dining")
            
with tab2:
    st.markdown("##### 🚕 Commute Options")
    m1, m2, m3 = st.columns(3)
    m1.metric("App Cabs", "₹350 - ₹700")
    m2.metric("Local Transit", "₹15 - ₹50")
    m3.metric("Train Station", "15 mins away")
    if st.button(f"📍 Open {Destination} Map"):
        st.write(f"Redirecting to live traffic map for {Destination}...")

with tab3:
    st.markdown(f"##### 🗺️ Explore {Destination}")
    col_a, col_b = st.columns(2)
    with col_a:
        st.checkbox("Historical Heritage Sites", value=True)
        st.checkbox("Main Shopping District")
    with col_b:
        st.checkbox("City Gardens & Parks")
        st.checkbox("Art & Culture Museum")

with tab4:
    st.markdown("##### 🏷️ Active Coupons")
    st.success("Code: FLYHIGH2026")
    st.caption("Get ₹1000 off on your next international flight.")
    st.warning("Code: TRIPSTAY10")
    st.caption("10% flat discount on selected hotels.")
