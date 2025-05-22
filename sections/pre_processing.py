import streamlit as st

def show_pre_processing():
    st.title("Nettoyage et Pré processing")
    tab1, tab2 = st.tabs(["Nettoyage", "Pré Processing"])
    with tab1:
        st.write("Traitement des valeurs manquantes, doublons, etc.")
    with tab2:
        st.write("Encodage, normalisation, etc.")