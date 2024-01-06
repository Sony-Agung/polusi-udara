import pickle
import streamlit as st
import pandas as pd

# Load the trained SVM model
with open('svm_model3.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

def preprocess_data(data):
    # Turn binary attributes into 0 and 1
    data['critical'] = data['critical'].map({'O3': 1, 'PM10': 2, 'SO2': 3, 'PM25': 4, 'CO': 5})
    return data

def get_air_quality_category(prediction):
    if prediction == 1:
        return "Baik"
    elif prediction == 2:
        return "Sedang"
    elif prediction == 3:
        return "Tidak Sehat"
    else:
        return "Unknown"

def main():
    st.title("Air Quality Prediction App")
    st.sidebar.header("User Input Features")

    # Collect user input features
    pm10 = st.sidebar.slider("PM10", min_value=0, max_value=100, value=30)
    so2 = st.sidebar.slider("SO2", min_value=0, max_value=100, value=22)
    co = st.sidebar.slider("CO", min_value=0, max_value=100, value=10)
    o3 = st.sidebar.slider("O3", min_value=0, max_value=100, value=22)
    no2 = st.sidebar.slider("NO2", min_value=0, max_value=100, value=11)
    max_val = st.sidebar.slider("Maximum", min_value=0, max_value=100, value=29)
    critical = st.sidebar.selectbox("Critical", ['O3', 'PM10', 'SO2', 'PM25', 'CO'], index=4)

    # Create a dictionary with user input
    user_input = {'pm10': pm10, 'so2': so2, 'co': co, 'o3': o3, 'no2': no2, 'max': max_val, 'critical': critical}

    # Convert the dictionary to a Pandas DataFrame
    input_data = pd.DataFrame([user_input])

    # Preprocess the user input data
    input_data = preprocess_data(input_data)

    # Make predictions
    if st.button("Predict"):
        prediction = classifier.predict(input_data)
        air_quality_category = get_air_quality_category(prediction[0])
        st.success(f"The predicted air quality category is: {air_quality_category}")

if __name__ == '__main__':
    main()
