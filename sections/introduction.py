import streamlit as st

def show_introduction():
    st.title("Introduction")
    st.subheader("Contexte du projet")
    st.markdown("""
    Ce projet a pour but de prédire si un client d'une banque va souscrire à un dépôt à terme suite à une campagne de marketing téléphonique.
    La banque souhaite améliorer ses campagnes marketing en ciblant les clients les plus susceptibles de souscrire à un dépôt à terme.
    """, unsafe_allow_html=True)

    st.subheader("Objectifs du projet")
    st.markdown("""
    - Analyser les données pour comprendre les facteurs influençant la souscription à un dépôt à terme.
    - Préparer les données pour la modélisation.
    - Construire des modèles de machine learning pour prédire la souscription à un dépôt à terme.
    - Évaluer les performances des modèles et choisir le meilleur.
    - Proposer des pistes d'amélioration continue pour le modèle.
    """, unsafe_allow_html=True)

    st.subheader("Jeu de données")
    st.markdown("""
    Nous allons utiliser un jeu de données contenant des informations sur les clients et les résultats de la campagne précédente.
    Le jeu de données est téléchargeable au lien suivant: https://www.kaggle.com/janiobachmann/bank-marketing-dataset
    """, unsafe_allow_html=True)