import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import h2o
from h2o.automl import H2OAutoML

# Load data
@st.cache
def load_data():
    path = "cancer patient data sets.csv"
    return pd.read_csv(path)

data = load_data()

# Set up Streamlit app
st.title('Cancer Patient Data Analysis')
st.write("""
## Overview of Cancer Risk Levels and Influencing Factors
This analysis explores the impact of lifestyle and environmental factors on cancer risk levels categorized as Low, Medium, or High. Our data encompasses a diverse range of predictors with a focus on four key areas:
- **Alcohol Use**
- **Smoking**
- **Occupational Hazards**
- **Air Pollution**
""")

# Dropdown for selecting the level
selected_level = st.selectbox("Select Level", options=data['Level'].unique())

# Filtered data based on selected level
filtered_data = data[data['Level'] == selected_level]

# Display histograms for selected features
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].hist(filtered_data['Air Pollution'], bins=10, color='skyblue')
ax[0, 0].set_title('Air Pollution')
ax[0, 1].hist(filtered_data['Alcohol use'], bins=10, color='lightgreen')
ax[0, 1].set_title('Alcohol use')
ax[1, 0].hist(filtered_data['Smoking'], bins=10, color='salmon')
ax[1, 0].set_title('Smoking')
ax[1, 1].hist(filtered_data['Occupational Hazards'], bins=10, color='orange')
ax[1, 1].set_title('Occupational Hazards')
plt.tight_layout()
st.pyplot(fig)

# Encode categorical data
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = label_encoder.fit_transform(data[column])

# Train a Ridge Regression model
X = data.drop('Level', axis=1)
y = data['Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_mse = mean_squared_error(y_test, y_pred)
st.write(f'Ridge Regression MSE: {ridge_mse}')

# Initialize H2O cluster
h2o.init()

# Convert pandas DataFrame to H2OFrame
data_h2o = H2OFrame(data)

# Configure H2O AutoML
aml = H2OAutoML(max_models=10, seed=1, project_name="Capstone_AutoML")

# Train models
aml.train(x=predictors, y=target, training_frame=train, leaderboard_frame=valid)

# View the AutoML Leaderboard
lb = aml.leaderboard
st.write(lb)

# Get the best model
best_model = aml.leader
st.write(best_model)

# Model performance metrics
mod_perf = best_model.model_performance(data)
st.write("Model Performance Metrics:")
st.write(mod_perf)
