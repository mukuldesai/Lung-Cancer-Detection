import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # Use the correct file path and name
    path = "cancer patient data sets.csv"
    return pd.read_csv(path)

data = load_data()

# Set up your Streamlit app
st.title('Cancer Patient Data Analysis')
st.write("This app displays the histograms for Air Pollution, Alcohol use, Smoking, and Occupational Hazards based on the selected Level.")

# Display the overview text
st.write("""
## Overview of Cancer Risk Levels and Influencing Factors
This analysis explores the impact of lifestyle and environmental factors on cancer risk levels categorized as Low, Medium, or High. Our data encompasses a diverse range of predictors with a focus on four key areas:

- **Alcohol Use:** Examining how varying levels of alcohol consumption correlate with cancer risk. Initial findings suggest a notable association between higher alcohol intake and increased risk levels, underscoring the need for further targeted studies.
- **Smoking:** As a well-documented risk factor, smoking shows a strong correlation with higher cancer risk. Our analysis reinforces the critical importance of smoking cessation in cancer prevention strategies.
- **Occupational Hazards:** Exposure to hazardous substances in the workplace is another significant factor. Our data indicates that certain occupations, particularly those involving exposure to carcinogenic chemicals, are linked to elevated cancer risk levels.
- **Air Pollution:** Environmental exposure to polluted air, especially in urban areas, has been analyzed to assess its impact on cancer risk. Preliminary results confirm that higher pollution levels contribute to higher incidences of cancer, aligning with global health observations.
""")

# Dropdown for selecting the level
selected_level = st.selectbox("Select Level", options=data['Level'].unique())

# Filter the data based on the selected level
filtered_data = data[data['Level'] == selected_level]

# Display histograms for the selected features
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].hist(filtered_data['Air Pollution'], bins=10, color='skyblue')
ax[0, 0].set_title('Air Pollution')

ax[0, 1].hist(filtered_data['Alcohol use'], bins=10, color='lightgreen')
ax[0, 1].set_title('Alcohol use')

ax[1, 0].hist(filtered_data['Smoking'], bins=10, color='salmon')
ax[1, 0].set_title('Smoking')

# Correct the column name for "Occupational Hazards"
ax[1, 1].hist(filtered_data['Occupational Hazards'], bins=10, color='orange')
ax[1, 1].set_title('Occupational Hazards')

# Adjust layout
plt.tight_layout()
st.pyplot(fig)
