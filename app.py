import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data with error handling and caching
@st.cache_data
def load_data():
    # Raw GitHub URL for the CSV file
    url = "https://raw.githubusercontent.com/mukuldesai/Lung-Cancer-Detection/main/cancer_patient_data_sets.csv"
    try:
        data = pd.read_csv(url)
        return data
    except FileNotFoundError:
        st.error("The file was not found. Please check the file path and try again.")
        return None
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please provide a valid CSV file.")
        return None
    except pd.errors.ParserError:
        st.error("Error parsing the file. Please ensure it is in the correct format.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# Load the data
data = load_data()

if data is not None:
    # Set up your Streamlit app
    st.title('Cancer Patient Data Analysis')
    st.write("This app displays the histograms for Air Pollution, Alcohol use, Smoking, and Occupational Hazards based on the selected cancer risk level.")

    # Overview section
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

    if not filtered_data.empty:
        # User input for number of bins
        num_bins = st.slider("Select Number of Bins", min_value=5, max_value=50, value=10)

        # Create histograms for key factors
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Air Pollution
        ax[0, 0].hist(filtered_data['Air Pollution'], bins=num_bins, color='skyblue', edgecolor='black')
        ax[0, 0].set_title('Air Pollution')
        ax[0, 0].set_xlabel('Air Pollution Level')
        ax[0, 0].set_ylabel('Frequency')

        # Alcohol use
        ax[0, 1].hist(filtered_data['Alcohol use'], bins=num_bins, color='lightgreen', edgecolor='black')
        ax[0, 1].set_title('Alcohol use')
        ax[0, 1].set_xlabel('Alcohol Use Level')
        ax[0, 1].set_ylabel('Frequency')

        # Smoking
        ax[1, 0].hist(filtered_data['Smoking'], bins=num_bins, color='salmon', edgecolor='black')
        ax[1, 0].set_title('Smoking')
        ax[1, 0].set_xlabel('Smoking Level')
        ax[1, 0].set_ylabel('Frequency')

        # Occupational Hazards
        ax[1, 1].hist(filtered_data['Occupational Hazards'], bins=num_bins, color='orange', edgecolor='black')
        ax[1, 1].set_title('Occupational Hazards')
        ax[1, 1].set_xlabel('Occupational Hazards Level')
        ax[1, 1].set_ylabel('Frequency')

        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write(f"No data available for the selected level: {selected_level}")

else:
    st.write("No data available for analysis.")
