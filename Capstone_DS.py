#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import shutil
import opendatasets as od
import pandas as pd

# Execute pip install command using subprocess
subprocess.run(['pip', 'install', 'opendatasets'])

# Set the directory containing kaggle.json
repo_path = '/mount/src/mukuldesai/demoDS/'  # Update this with your actual path
kaggle_credentials_dir = os.path.join(repo_path, 'kaggle.json')


# Move kaggle.json to the appropriate directory
shutil.move(kaggle_credentials_dir, os.path.expanduser('~/.kaggle/kaggle.json'))

# Dataset URL
data = "https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link"

# Download dataset using opendatasets
od.download(data)

# Define the directory containing the downloaded dataset
data_dir = './cancer-patients-and-air-pollution-a-new-link'

# Check if the data directory exists
if os.path.exists(data_dir):
    # List files in the data directory
    files = os.listdir(data_dir)
    print("Contents of data directory:", files)
    
    # Read the CSV file
    data_file_path = os.path.join(data_dir, 'cancer patient data sets.csv')
    data = pd.read_csv(data_file_path)
    print("Loaded dataset successfully!")
else:
    print("The data directory does not exist or is not accessible.")



# In[6]:



# In[7]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[8]:





data_dir = '.\cancer-patients-and-air-pollution-a-new-link'


# In[10]:


os.listdir(data_dir)


# In[11]:


data = pd.read_csv('.\cancer-patients-and-air-pollution-a-new-link\cancer patient data sets.csv')


# In[12]:


data.head()


# In[13]:


# Displaying count of null values
data.isnull().sum()


# In[14]:


# Displaying count of null values
data.isnull().sum()


# Since no row data contains the null values so none of the rows have been deleted from dataset

# In[15]:


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


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[20]:


# Analyze specific columns like 'Age', 'Gender', 'Air Pollution', etc.
# You can plot histograms, bar charts, or other visualizations to understand distributions
import seaborn as sns
import matplotlib.pyplot as plt

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


# In[31]:


from sklearn import preprocessing

# Create x to store scaled values as floats
x = featured_dataset[['Level', 'Smoking']].values.astype(float)

# Initialize MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()

# Transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
featured_dataset[['Level', 'Smoking']] = pd.DataFrame(x_scaled)



# In[32]:


plt.figure(figsize=(40,7))
sns.boxplot(data=featured_dataset)


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming 'data' is your DataFrame and it's already defined.
# Make sure to import the necessary libraries if you haven't already:
# import pandas as pd
# import numpy as np

logisticRegrNoOutliers = LogisticRegression()

# Prepare your features (X) and target (y) datasets properly.
# Exclude the target variable 'Level' from the features dataset.
X_no_outlier_data = data.drop(columns=['Level'])
y_no_outlier_data = data['Level'].values.ravel()  # Flatten the array if y is in DataFrame format

# Split the dataset into training and test sets.
X_train_no_outlier_data, X_test_no_outlier_data, y_train_no_outlier_data, y_test_no_outlier_data = train_test_split(
    X_no_outlier_data, y_no_outlier_data, test_size=0.2, random_state=42, stratify=y_no_outlier_data
)

# Fit the Logistic Regression model on the training data.
logisticRegrNoOutliers.fit(X_train_no_outlier_data, y_train_no_outlier_data)

# Predict using the test dataset.
predictions = logisticRegrNoOutliers.predict(X_test_no_outlier_data)
predictions


# **Now we have to perform AutoML on this **

# In[34]:


import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Add a constant term for the intercept
X = add_constant(data.drop('Level', axis=1))

# Create a DataFrame to hold feature names and their VIF values
vif = pd.DataFrame()
vif["Feature"] = X.columns
vif["VIF Value"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF values
print(vif)


# The presence of multicollinearity suggests that the independent variables are not completely independent from each other. It becomes difficult to determine the individual effect of one predictor on the target variable because it is so closely related to one or more of the other predictors.
# 
# Alcohol use: With a VIF of 17.61, it suggests a high level of multicollinearity with one or more other independent variables.
# 
# Occupational Hazards: Has a VIF of 21.31, which is quite high and indicates significant multicollinearity.
# 
# Genetic Risk: Also has a high VIF of 22.85, suggesting that this variable is linearly related to other variables in the dataset.
# 
# Chronic Lung Disease, Balanced Diet, Obesity, Passive Smoker, Chest Pain, Coughing of Blood: All have VIFs above 5, indicating moderate to high multicollinearity.
# 
# Level_encoded: Although this is your encoded target variable and typically wouldn't be included in the VIF calculation for predictors, a VIF of 13.82 would be considered high if it were a predictor.

# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define your features and target variable
X = data.drop('Level', axis=1)  # Assuming 'Level' is your target variable
y = data['Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge Regression model with an alpha value
ridge_model = Ridge(alpha=1.0)  # Alpha is the regularization strength

# Fit the model on the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = ridge_model.predict(X_test)

# Evaluate the model
ridge_mse = mean_squared_error(y_test, y_pred)
print(f'Ridge Regression MSE: {ridge_mse}')


# **H2O**
# 

# In[36]:




# In[37]:


import h2o
h2o.init()  # This will start an H2O cluster


# In[38]:


# Ensure your pandas DataFrame is defined, e.g., 'data'
data_h2o = h2o.H2OFrame(data)


# In[39]:


data = h2o.H2OFrame(data)


# In[40]:





# In[41]:


import h2o
from h2o.frame import H2OFrame
import pandas as pd

# Assuming 'df' is your pandas DataFrame containing the data
data = pd.read_csv('.\cancer-patients-and-air-pollution-a-new-link\cancer patient data sets.csv')

# Initialize H2O cluster
h2o.init()

# Convert pandas DataFrame to H2OFrame
data = H2OFrame(data)

# Now 'data' should exist in the H2O cluster
print("H2OFrame 'data' has been created.")


# In[42]:


train, test = data.split_frame(seed = 1234, destination_frames = ["train.hex", "test.hex"])


# In[43]:


# check the number of train set and test set
train["is_train"] = 1
test["is_train"] = 0

drift_data = train.rbind(test)
drift_data["is_train"] = drift_data["is_train"].asfactor()


# In[44]:


drift_data["is_train"].table()


# **Start to train in 10 models by AutoML**

# In[45]:


from h2o.automl import H2OAutoML


# In[46]:


# Convert to H2OFrame
data_h2o = h2o.H2OFrame(data)

# Split the data into training and validation sets
train, valid = data_h2o.split_frame(ratios=[0.8], seed=42)


# In[47]:


target = 'Level'
predictors = train.columns
predictors.remove(target)

# Configure H2O AutoML
aml = H2OAutoML(max_models=10, seed=1, project_name="Capstone_AutoML")

# Train models
aml.train(x=predictors, y=target, training_frame=train, leaderboard_frame=valid)


# In[48]:


# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)


# In[49]:


best_model = aml.leader
best_model.model_performance(test)


# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
best_model.varimp_plot()


# In[51]:


# Create a 80/20 train/test split
pct_rows=0.80
data_train, data_test = data.split_frame([pct_rows])


# In[52]:


#after split check rows and columns
print(data_train.shape)
print(data_test.shape)


# Effective Model Selection: The AutoML process successfully navigated through a variety of machine learning algorithms and determined the best-performing model tailored to the complexity and structure of the dataset, which contains a mix of demographic, environmental, and health-related factors for predicting cancer risk levels.
# 
# Hyperparameter Optimization: Through systematic hyperparameter tuning, AutoML identified optimal settings that enhanced model performance. This optimization balanced model complexity and generalization, leading to improved accuracy while preventing overfitting.
# 
# Multicollinearity and Regularization: The assignment highlighted the presence of multicollinearity in the dataset. The application of regularization techniques within the AutoML models helped mitigate its impact, leading to more reliable and interpretable coefficient estimates.
# 
# Model Validation and Practicality: The final model was rigorously evaluated against a hold-out test set, ensuring its predictive validity. The low MSE scores suggest that the model's predictions are highly accurate, making it a potentially useful tool for medical practitioners to assess cancer risk levels based on a range of patient information.
# 
# In summary, this AutoML assignment demonstrated the capability of automated tools to streamline the development of predictive models, handle complex datasets, and provide insights that are both statistically significant and potentially valuable in a real-world clinical setting.

# In[53]:


#data manupulation
import pandas as pd
#numerical combination
import numpy as np 
#plotting data and create visualization
import matplotlib.pyplot as plt           
import seaborn as sns
import plotly.express as px
import graphviz

from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.tree import plot_tree
import pydotplus #pip install pydotplus
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
from xgboost import plot_importance



# In[54]:


if best_model.algo in ['gbm','drf','xrt','xgboost']:
  best_model.varimp_plot()


# In[55]:


# Assuming glm_index is defined and assigned somewhere in your code
glm_index = 0  # Assigning a value to glm_index

# Now you can use glm_index in conditional statements
if glm_index != 0:
    print(glm_index)
    glm_model = h2o.get_model(auml.leaderboard[glm_index, 'model_id'])
    print(glm_model.algo) 
    glm_model.std_coef_plot()

# Now describe the data regardless of the condition
data.describe()


# In[56]:


if glm_index is not 0:
  print(glm_index)
  glm_model=h2o.get_model(auml.leaderboard[glm_index,'model_id'])
  print(glm_model.algo) 
  glm_model.std_coef_plot()


# In[57]:


print(best_model.auc(train = True))


# In[58]:


def model_performance_stats(perf):
    d={}
    try:    
      d['mse']=perf.mse()
    except:
      pass      
    try:    
      d['rmse']=perf.rmse() 
    except:
      pass      
    try:    
      d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
      pass      
    try:    
      d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
      pass      
    try:    
      d['residual_deviance']=perf.residual_deviance() 
    except:
      pass      
    try:    
      d['null_deviance']=perf.null_deviance() 
    except:
      pass      
    try:    
      d['aic']=perf.aic() 
    except:
      pass      
    try:
      d['logloss']=perf.logloss() 
    except:
      pass    
    try:
      d['auc']=perf.auc()
    except:
      pass  
    try:
      d['gini']=perf.gini()
    except:
      pass    
    return d


# In[59]:


mod_perf=best_model.model_performance(data)
stats_test={}
stats_test=model_performance_stats(mod_perf)
stats_test


# In[63]:


model_index=0
glm_index=0
glm_model=''
aml_leaderboard_df=aml.leaderboard.as_data_frame()
models_dict={}
for m in aml_leaderboard_df['model_id']:
  models_dict[m]=model_index
  if 'StackedEnsemble' not in m:
    break 
  model_index=model_index+1  

for m in aml_leaderboard_df['model_id']:
  if 'GLM' in m:
    models_dict[m]=glm_index
    break  
  glm_index=glm_index+1     
models_dict


# In[65]:


print(model_index)
best_model = h2o.get_model(aml.leaderboard[model_index,'model_id'])


# In[66]:


best_model.algo


# In[68]:





# In[ ]:




