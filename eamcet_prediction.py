import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv("College_data.csv")

label_encoder = LabelEncoder()
def college_prediction(input_data):
    input1=input_data[1:]
    input1[0]=label_encoder.fit_transform(input1[0])
    input1[1]=label_encoder.fit_transform(input1[1])
    input_data=input_data[0]+input1
    input_data = [float(x) for x in input_data]
    input_data_modified = np.asarray(input_data)
    input_data_reshaped = input_data_modified.reshape(1, -1)
    
    data['Caste'] = label_encoder.fit_transform(data['Caste'])
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['College_Branch'] = label_encoder.fit_transform(data['College_Branch'])

    # Feature selection
    X = data[['Rank', 'Caste', 'Gender']]
    y = data['College_Branch']

    # Train  model
    dt = DecisionTreeClassifier() 
    dt.fit(X, y)
    
    probabilities = dt.predict_proba(input_data_reshaped)
    sorted_predictions = (-probabilities).argsort(axis=1)[:, :10]  # Select top 10
    
    st.write("Top 10 Predictions:")
    for i, prediction in enumerate(sorted_predictions):
        st.write(f"Sample {i + 1}:")
        for j, idx in enumerate(prediction):
            st.write(f"{j + 1}. {dt.classes_[idx]} - Probability: {probabilities[i][idx]}")

def main():
    st.title("Engineering College and Branch Prediction")
    
    Rank = st.text_input("EAMCET Rank:", value='', max_chars=None, key=None, type='default')
    Caste = st.text_input("Caste (e.g., BC_D):", value='', max_chars=None, key=None, type='default')
    Gender = st.text_input("Gender (M or F):", value='', max_chars=None, key=None, type='default')
    
    if st.button("Predict"):
        if not Rank or not Caste or not Gender:
            st.error("Please provide all input values.")
        else:
            input_data = [Rank, Caste, Gender]
            college_prediction(input_data)

if __name__ == '__main__':
    main()
