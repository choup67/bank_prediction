import streamlit as st

def show_conclusion():
    st.title("Conclusion")
    tab1, tab2 = st.tabs(["Observations", "Améliorations continues"])
    with tab1:
        st.write("Résultats observés, conclusions principales")
    with tab2:
        st.write("Pistes d'amélioration du modèle et du pipeline")