import streamlit as st

def show_exploration():
    st.title("Exploration et visualisation des données")
    tab1, tab2 = st.tabs(["Visualisations catégorielles", "Visualisations numériques"])
    with tab1:
        st.write("Graphiques des variables catégorielles")
    with tab2:
        st.write("Graphiques des variables numériques")