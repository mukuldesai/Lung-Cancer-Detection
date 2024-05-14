import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

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

ax[1, 1].hist(filtered_data['Occupational Hazards'], bins=10, color='orange')
ax[1, 1].set_title('Occupational Hazards')

# Adjust layout
plt.tight_layout()
st.pyplot(fig)

#Checking the data types of the columns in the dataset
column_data_types = data.dtypes

st.write("Data Types of Columns:")
st.write(column_data_types)

# Changing the Level data for better analysis of the data
data["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
data['Level'] = data['Level'].astype('int64')
st.write('CancerLevel:', data['Level'].unique())

# Display the shape of the dataset (number of rows and columns)
st.write("Shape of the dataset:", data.shape)

# Display the data types of each column
st.write("Data types of each column:")
st.write(data.dtypes)

# Example: Histogram of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
st.pyplot()

# Histogram of Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=data)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
st.pyplot()

# Encode categorical data
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
st.pyplot()

# Checking the distribution of Independent variables
field_names = data[["Age", "Alcohol use", "Genetic Risk", "Frequent Cold", "Dry Cough", "chronic Lung Disease", "Smoking", "Passive Smoker"]]
for column in field_names.columns:
    sns.set(rc={"figure.figsize": (8, 4)})
    sns.histplot(data[column])
    st.pyplot()

# Remove unnecessary data columns
data.drop(['index', 'Patient Id'], inplace=True, axis=1)

# Boxplot for data ranges
plt.figure(figsize=(40, 6))
sns.boxplot(data=data)
st.pyplot()

# Correlation matrix among columns
corr = data.corr()
plt.figure(figsize=(20, 12))
sns.heatmap(corr, annot=True)
plt.title('Correlation Matrix')
st.pyplot()

# Pairplot for colinearity check
sns.pairplot(data)
st.pyplot()

# Dependency correlation with Level column
st.write("Dependency correlation with Level column:")
st.write(data.corr()['Level'].sort_values(ascending=False))

# Using OLS for finding the p value to check the significant features
model = sm.OLS(data['Level'], data[['Air Pollution', 'Smoking', 'Passive Smoker']]).fit()
# Print out the statistics
st.write("OLS model summary:")
st.write(model.summary())

# Select the features for analysis and outlier detection
featured_dataset = data[['Air Pollution', 'Smoking', 'Passive Smoker', 'Level']]
fields = featured_dataset.columns
# Iterate over each feature column and create a boxplot
for column in fields:
    sns.boxplot(data=featured_dataset[column])
    plt.title(column)
    st.pyplot()
