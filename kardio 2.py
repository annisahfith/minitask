import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
with open(r'D:\MBKM\SKILVUL\PYTHON\Kardio\gb_model.pkl', 'rb') as file:
    kardio_model = pickle.load(file)

# Load the scaler
with open(r'D:\MBKM\SKILVUL\PYTHON\Kardio\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

# Function to preprocess user input
def preprocess_input(user_input):
    # Extract user input
    age = user_input["age_years"]
    bmi = calculate_bmi(user_input["weight"], user_input["height"])
    systolic_blood_pressure = user_input["systolic_blood_pressure"]
    diastolic_blood_pressure = user_input["diastolic_blood_pressure"]
    
    # Prepare the array for scaling
    input_arr_to_scale = np.array([[age, bmi, systolic_blood_pressure, diastolic_blood_pressure]])
    scaled_arr = scaler.transform(input_arr_to_scale)
    
    # Convert categorical input to numerical
    gender_dict = {"Pria": 1, "Wanita": 2}
    cholesterol_dict = {"Normal": 1, "Di atas normal": 2, "Jauh di atas normal": 3}
    glucose_dict = {"Normal": 1, "Di atas normal": 2, "Jauh di atas normal": 3}
    binary_dict = {"Tidak": 0, "Ya": 1}
    
    gender = gender_dict[user_input["gender"]]
    cholesterol_level = cholesterol_dict[user_input["cholesterol_level"]]
    glucose_level = glucose_dict[user_input["glucose_level"]]
    smoke = binary_dict[user_input["smoke"]]
    alcohol_intake = binary_dict[user_input["alcohol_intake"]]
    active = binary_dict[user_input["active"]]
    
    # Combine scaled and unscaled inputs
    input_arr = np.hstack((scaled_arr, [[gender, cholesterol_level, glucose_level, smoke, alcohol_intake, active]]))
    
    return input_arr

# Function to make prediction
def predict_kardio_disease(user_input):
    prediction = kardio_model.predict(user_input)
    return prediction

# Streamlit app
st.set_page_config(page_title="Prediksi Penyakit Kardiovaskuler", layout="centered")

# Header
st.title('Prediksi Penyakit Kardiovaskuler')
st.markdown("""
    Aplikasi ini bertujuan untuk memprediksi kemungkinan seseorang terkena penyakit kardiovaskuler
    berdasarkan beberapa parameter kesehatan. Masukkan data Anda di bawah ini untuk melihat hasil prediksinya.
""")

# Form for user input
st.markdown("## **Masukkan Data Anda**")
input_card_1, input_card_2 = st.columns(2)

with input_card_1:
    st.markdown("### **Data Pribadi**")
    gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
    age_years = st.number_input('Usia', min_value=0, max_value=90, value=40, step=1)
    weight = st.number_input('Berat Badan (kg)', min_value=0.0, value=60.0, format="%.1f")
    height = st.number_input('Tinggi Badan (cm)', min_value=0.0, value=170.0, format="%.1f")

with input_card_2:
    st.markdown("### **Data Kesehatan**")
    systolic_blood_pressure = st.number_input('Tekanan Darah Sistolik', min_value=0, max_value=300, value=120, step=1)
    diastolic_blood_pressure = st.number_input('Tekanan Darah Diastolik', min_value=0, max_value=300, value=80, step=1)
    cholesterol_level_input = st.selectbox("Tingkat Kolesterol", ["Normal", "Di atas normal", "Jauh di atas normal"])
    glucose_level_input = st.selectbox("Tingkat Glukosa", ["Normal", "Di atas normal", "Jauh di atas normal"])

# Additional inputs in three columns
st.markdown("### **Kebiasaan dan Aktivitas**")
col1, col2, col3 = st.columns(3)
with col1:
    smoke_input = st.radio("Merokok", ("Tidak", "Ya"))

with col2:
    alcohol_intake_input = st.radio("Konsumsi Alkohol", ("Tidak", "Ya"))

with col3:
    active_input = st.radio("Aktif Berolahraga", ("Tidak", "Ya"))

# Make prediction when button is clicked
if st.button('Test Prediksi'):
    user_input = {
        "gender": gender,
        "age_years": age_years,
        "weight": weight,
        "height": height,
        "systolic_blood_pressure": systolic_blood_pressure,
        "diastolic_blood_pressure": diastolic_blood_pressure,
        "cholesterol_level": cholesterol_level_input,
        "glucose_level": glucose_level_input,
        "smoke": smoke_input,
        "alcohol_intake": alcohol_intake_input,
        "active": active_input
    }
    
    preprocessed_input = preprocess_input(user_input)
    prediction_result = predict_kardio_disease(preprocessed_input)
    st.write("Prediction Probabilities:", prediction_result)
    if prediction_result[0] == 1:
        st.markdown("### Hasil Prediksi")
        st.success('Pasien terkena Kardiovaskular')
    else:
        st.markdown("### Hasil Prediksi")
        st.success('Pasien tidak terkena Kardiovaskular')

