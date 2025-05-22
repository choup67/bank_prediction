import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "bank.csv")
    df = pd.read_csv(file_path, sep = ',')
    return df