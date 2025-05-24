from utils.data_functions import prepa_num, prepa_cat, replace_unknown, create_active_loan, transform_to_bool, remove_useless_col
import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), "..", "bank.csv")
    df = pd.read_csv(file_path, sep = ',')
    return df

@st.cache_data
def load_and_prepare_data():
    df = load_data()
    df = prepa_num(df)
    df = prepa_cat(df)
    return df

@st.cache_data
def load_cleaned_and_prepared_data():
    df = load_and_prepare_data()
    df = replace_unknown(df)
    df = transform_to_bool(df)
    df = create_active_loan(df)
    df = remove_useless_col(df)
    return df

