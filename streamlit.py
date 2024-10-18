import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
def load_data():
    path = "cancer patient data sets.csv"  # Ensure the correct path
    return pd.read_csv(path)

data = load_data()

# Set up your Streamlit app
st.title('Cancer Patient Data Analysis')
st.write("This app displays the histograms for Air Pollution, Alcohol use, Smoking, and OccuPational Hazards based on the selected Level.")

# Overview section
st.write("""
## Overview of Cancer Risk Levels and Influencing Factors
This analysis explores the impact of lifestyle and environmental factors on cancer risk levels categorized as Low, Medium, or High. Our data encompasses a diverse range of predictors with a focus on four key areas:

- **Alcohol Use:** Examining how varying levels of alcohol consumption correlate with cancer risk.
- **Smoking:** As a well-documented risk factor, smoking shows a strong correlation with higher cancer risk.
- **Occupational Hazards:** Exposure to hazardous substances in the workplace is another significant factor.
- **Air Pollution:** Environmental exposure to polluted air has been analyzed to assess its impact on cancer risk.
""")

# Dropdown for selecting cancer risk level
selected_level = st.selectbox("Select Level", options=data['Level'].unique())

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

# OccuPational Hazards (using the correct column name)
ax[1, 1].hist(filtered_data['OccuPational Hazards'], bins=10, color='orange')
ax[1, 1].set_title('OccuPational Hazards')

# Adjust layout and display the plot
plt.tight_layout()
st.pyplot(fig)
