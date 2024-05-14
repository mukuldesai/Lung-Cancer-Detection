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

ax[1, 1].hist(filtered_data['OccuPational Hazards'], bins=10, color='orange')
ax[1, 1].set_title('Occupational Hazards')

# Adjust layout
plt.tight_layout()
st.pyplot(fig)

according to my other code

#Checking the data types of the columns in the dataset
column_data_types = data.dtypes


# In[16]:


column_data_types


# This String and Integer are the datatypes, that means the numerical and categorical

# In[17]:


# Changing the Level data for better analysis of the data
data["Level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
data['Level'] = data['Level'].astype('int64')
print('CancerLevel: ', data['Level'].unique())


# In[18]:


# Display the shape of the dataset (number of rows and columns)
print("Shape of the dataset:", data.shape)

# Display the data types of each column
print("\nData types of each column:")
print(data.dtypes)



# In[19]:





# In[20]:


# Analyze specific columns like 'Age', 'Gender', 'Air Pollution', etc.
# You can plot histograms, bar charts, or other visualizations to understand distributions
import seaborn as sns
import matplotlib.pyplot as plt


data = data.replace([float('inf'), float('-inf')], pd.NA).dropna(subset=['Age'])

# Example: Histogram of Age
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Import the necessary libraries for data visualization: seaborn (as sns) and matplotlib.pyplot (as plt).
# Create a histogram of the 'Age' column in the data DataFrame, specifying 20 bins and enabling kernel density estimation (kde).

# In[21]:


# Histogram of Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=data)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[22]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Iterate over each column
for column in data.columns:
    # Check if the column contains non-numeric data
    if data[column].dtype == 'object':
        # Encode non-numeric data to numeric values
        data[column] = label_encoder.fit_transform(data[column])

# Now you can compute the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# Import the LabelEncoder class from the sklearn.preprocessing module.
# Initialize a LabelEncoder object.
# Iterate over each column in the DataFrame (data).
# Check if the data type of the column is non-numeric (i.e., 'object' type).
# If the column contains non-numeric data, use the fit_transform() method of the LabelEncoder object to encode the non-numeric data into numeric values.
# Once all non-numeric data has been encoded, display the correlation matrix of the DataFrame using sns.heatmap(), which creates a heatmap visualization with correlation values annotated on it.

# In[23]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Checking the distribution of Independent variables
field_names = data[[
    "Age", "Alcohol use", "Genetic Risk", "Frequent Cold", "Dry Cough", "chronic Lung Disease", "Smoking", "Passive Smoker"
]]

for column in field_names.columns:
    sns.set(rc={"figure.figsize": (8, 4)})
    sns.histplot(data[column])
    plt.show()


# In[24]:


#Remove the data columns that are not required for analysis as they have no significant value
data.drop(['index', 'Patient Id'], inplace=True, axis=1)


# In[25]:


#Checking the Ranges of the predictor variables and dependent variable
plt.figure(figsize=(40,6))
sns.boxplot(data=data)


# In[26]:


# Compute the correlation matrix among the columns of the dataset
corr = data.corr()

# Set the size of the heatmap figure
plt.figure(figsize=(20,12))

# Create a heatmap to visualize the correlation matrix with annotations
# annot=True displays the correlation values on the heatmap
sns.heatmap(corr, annot=True)

# Display the heatmap
plt.show()


# In[27]:


# Now we check the colinearity between the columns
sns.pairplot(data)


# In[28]:


# Dependency correlation with Level column
data.corr()['Level'].sort_values(ascending=False)


# In[29]:


#Using OLS for finding the p value to check the significant features
import statsmodels.api as sm

model = sm.OLS(data['Level'], data[['Air Pollution',  'Smoking', 'Passive Smoker']]).fit()

# Print out the statistics
model.summary()


# In[30]:


# Select the features for analysis and outlier detection
featured_dataset = data[['Air Pollution', 'Smoking', 'Passive Smoker', 'Level']]

# Get the column names of the selected features
fields = featured_dataset.columns

# Iterate over each feature column and create a boxplot
for column in fields:
    # Create a boxplot for the current feature
    sns.boxplot(data=featured_dataset[column])
    
    # Set the title of the boxplot as the name of the feature
    plt.title(column)
    
    # Display the boxplot
    plt.show()

