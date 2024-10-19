import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data with error handling and caching for better performance
@st.cache_data
def load_data():
    try:
        path = "cancer_patient_data_sets.csv"  # Ensure the correct path to the CSV file
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error("The file was not found. Please check the file path and try again.")
        return None

data = load_data()

if data is not None:
    # Set up your Streamlit app
    st.title('Cancer Patient Data Analysis')
    st.write("This app displays the histograms for Air Pollution, Alcohol use, Smoking, and Occupational Hazards based on the selected cancer risk level.")

    # Overview section with a more concise description
    st.write("""
    ## Overview of Cancer Risk Levels and Influencing Factors
    Explore the impact of lifestyle and environmental factors on cancer risk levels categorized as Low, Medium, or High:
    - **Alcohol Use**: Examines the correlation between alcohol consumption and cancer risk.
    - **Smoking**: Analyzes the well-documented relationship between smoking and cancer.
    - **Occupational Hazards**: Looks at workplace exposure to hazardous substances.
    - **Air Pollution**: Studies environmental exposure to polluted air and its impact.
    """)

    # Dropdown for selecting cancer risk level
    selected_level = st.selectbox("Select Cancer Risk Level", options=data['Level'].unique())

    # Filter the data based on the selected level
    filtered_data = data[data['Level'] == selected_level]

    # Create and display histograms for key factors
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Air Pollution
    ax[0, 0].hist(filtered_data['Air Pollution'], bins=10, color='skyblue')
    ax[0, 0].set_title('Air Pollution')

    # Alcohol use
    ax[0, 1].hist(filtered_data['Alcohol use'], bins=10, color='lightgreen')
    ax[0, 1].set_title('Alcohol use')

    # Smoking
    ax[1, 0].hist(filtered_data['Smoking'], bins=10, color='salmon')
    ax[1, 0].set_title('Smoking')

    # Occupational Hazards (ensure the correct column name)
    ax[1, 1].hist(filtered_data['Occupational Hazards'], bins=10, color='orange')
    ax[1, 1].set_title('Occupational Hazards')

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.write("No data available for analysis.")
