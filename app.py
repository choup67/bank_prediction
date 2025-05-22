import streamlit as st

from sections.introduction import show_introduction
from sections.decouverte import show_decouverte
from sections.exploration import show_exploration
from sections.pre_processing import show_pre_processing
from sections.modelisation import show_modelisation
from sections.conclusion import show_conclusion

st.set_page_config(page_title = "Projet ML", layout = "wide")

section = st.sidebar.radio("Navigation", [
    "Introduction",
    "Découverte des données",
    "Exploration et visualisation des données",
    "Nettoyage et Pré processing",
    "Modélisation",
    "Conclusion"
])

if section == "Introduction":
    show_introduction()

elif section == "Découverte des données":
    show_decouverte()

elif section == "Exploration et visualisation des données":
    show_exploration()

elif section == "Nettoyage et Pré processing":
    show_pre_processing()

elif section == "Modélisation":
    show_modelisation()

elif section == "Conclusion":
    show_conclusion()