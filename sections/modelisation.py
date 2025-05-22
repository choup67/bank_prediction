import streamlit as st

def show_modelisation():
    st.title("Modélisation")
    tab1, tab2 = st.tabs(["Premières itérations et interprétation", "Optimisations"])
    with tab1:
        st.write("Premiers modèles, scores, importance des variables")
    with tab2:
        st.write("Optimisation par grid search, random search, etc.")