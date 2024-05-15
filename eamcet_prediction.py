import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def map_gender(gender):
    return 1 if gender == 'M' else 0

def map_caste(caste):
    caste_mapping = {'BC_A': 0, 'BC_B': 1, 'BC_C': 2, 'BC_D': 3, 'BC_E': 4, 'OC': 5, 'SC': 6, 'ST': 7}
    return caste_mapping.get(caste, -1)  
    
label_encoder = LabelEncoder()

def college_prediction(input_data, top_n=10):
    try:
        # Map gender and caste selections to numerical values
        input_data[2] = map_gender(input_data[2])
        input_data[1] = map_caste(input_data[1])

        # Convert input data to numpy array and reshape
        input_data_modified = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_modified.reshape(1, -1)
        
        # Predict the probabilities of each class
        probabilities = loaded_model.predict_proba(input_data_reshaped)
        
        # Get the indices of the top predicted classes
        top_indices = np.argsort(probabilities[0])[::-1][:top_n]
        
        # Display the top predictions
        st.write(f"Top {top_n} Predictions:")
        for i, idx in enumerate(top_indices):
            st.write(f"{i + 1}. {loaded_model.classes_[idx]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


def main():
    st.title("Engineering College and Branch Prediction")
    # Load data
    data = pd.read_csv("College_data.csv")
    data['Caste'] = label_encoder.fit_transform(data['Caste'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['College_Branch'] = label_encoder.fit_transform(data['College_Branch'])
    
    # Feature selection
    X = data[['Rank', 'Caste', 'Gender']]
    y = data['College_Branch']
    # Train  model
    dt = DecisionTreeClassifier() 
    dt.fit(X, y)

    Rank = st.text_input("EAMCET Rank:")
    Caste = st.selectbox("Caste", ['None','BC_A', 'BC_B', 'BC_C', 'BC_D', 'BC_E', 'OC', 'SC', 'ST'])
    Gender = st.selectbox("Gender", ['None','M', 'F'])
    
    # Checkbox for selecting the number of top predictions to display
    top_n_values = [1, 5, 10, 20, 50, 100]
    top_n = st.selectbox("Select Top N Predictions:", top_n_values, index=2)
    
    if st.button("Predict"):
        if not Rank or not Caste or not Gender:
            st.error("Please provide all input values.")
        else:
            input_data = [Rank, Caste, Gender]
            college_prediction(input_data, top_n)

if __name__ == '__main__':
    main()
