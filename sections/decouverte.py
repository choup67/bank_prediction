import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data

def show_decouverte():
    # Chargement des données
    df = load_data()

    st.title("Découverte des données")
    tab1, tab2, tab3 = st.tabs(["Aperçu des données", "Analyse catégorielles", "Analyse numériques"])
    with tab1:
        st.subheader("Dimensions du jeu de données")
        # Dimensions du jeu de données
        st.markdown(f"Le dataset contient **{df.shape[0]} lignes** et **{df.shape[1]} colonnes**.")

        # Aperçu du dataset
        st.subheader("Aperçu des premières lignes")
        st.dataframe(df.head())

        # Types de variables
        st.subheader("Types des variables")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Variable", 0: "Type"}))

        # Valeurs manquantes
        st.subheader("Valeurs manquantes")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing.reset_index().rename(columns={"index": "Variable", 0: "Valeurs manquantes"}))
        else:
            st.success("Aucune valeur manquante détectée.")

        # Doublons
        st.subheader("Doublons")
        duplicate = df.duplicated().sum()
        duplicate = duplicate[duplicate > 0]
        if (duplicate > 0).any():
            st.warning(f"Il y a **{duplicate} doublons** dans le dataset.")
        else:
            st.success("Aucune doublon détecté.")

    with tab2:
       
        var_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']

        st.header("Analyse des variables catégorielles")

        # Liste déroulante pour choisir une variable catégorielle
        selected_cat = st.selectbox("Choisissez une variable catégorielle :", var_cat)

        # Création du DataFrame de statistiques
        df_stats = pd.DataFrame({
            'count': df[selected_cat].value_counts(),
            'percentage': df[selected_cat].value_counts(normalize = True) * 100
        })

        # Nettoyage du DataFrame
        df_stats.reset_index(inplace = True)
        df_stats.columns = ['modalite', 'count', 'percentage']
        df_stats['percentage'] = df_stats['percentage'].round(2)

        # Affichage des résultats
        nb_cat = df[selected_cat].nunique()
        st.subheader(f"Distribution de la variable : {selected_cat} ({nb_cat} modalités possibles)")
        st.dataframe(df_stats)

        st.markdown("""
        ### Remarques générales

        > On remarque que les variables ont le plus souvent entre 2 et 4 modalités.  
        > Sauf pour la variable `"job"` et évidemment les mois.  
        
        > On peut aussi noter l’apparition de la valeur `"unknown"` pour certaines variables : `job`, `education`, `contact` et `poutcome`.  
        > Ce qui laisse sous-entendre que même si on n’avait pas de valeurs nulles dans le dataset, il y a tout de même des données manquantes pour certaines catégories.  
                    
        > Dans certains cas, l’impact est faible car peu de données manquantes,  
        > mais pour d’autres variables comme `"poutcome"`, la majorité des données sont `"unknown"`.  
        > Il faudra donc garder ces informations en tête pour la suite du projet et décider quoi faire avec ces variables et données manquantes  
        > (exemple : réduire le dataset).  

        > Certaines variables contiennent les valeurs `"no"` et `"yes"`.  
        > Comme ces variables ne contiennent pas de valeurs manquantes,  
        > il peut être judicieux de les transformer en type booléen.  
        > Cela permettra d’optimiser les calculs, la mémoire et la manipulation des données.  
        """, unsafe_allow_html = True)

        st.subheader("Détail des variables catégorielles")
        if st.checkbox("Afficher les observations sur les variables catégorielles", value = True):
            st.markdown("""
        - `job`  
            - Type ou catégorie d'emploi  
            - Contient des valeurs manquantes  
        <br>
        - `marital`  
            - Status marital  
            - Pas de valeurs manquantes  
        <br>
        - `education`  
            - Niveau d'études  
            - Contient des valeurs manquantes  
        <br>
        - `default`  
            - Permet de savoir si le client a déjà fait défaut sur un crédit (yes / no)  
            - Pas de valeurs manquantes  
            - On est en présence d'une variable qui ne prend que deux valeurs yes ou no. On pourrait donc potentiellement passer la variable en Booléen  
        <br>
        - `housing`  
            - Permet de savoir si le client a un prêt immobilier en cours (yes / no)  
            - Pas de valeurs manquantes  
            - On est en présence d'une variable qui ne prend que deux valeurs yes ou no. On pourrait donc potentiellement passer la variable en Booléen  
        <br>
        - `loan`  
            - Permet de savoir si le client a un prêt personnel en cours (yes / no)  
            - Pas de valeurs manquantes  
            - On est en présence d'une variable qui ne prend que deux valeurs yes ou no.  
                On pourrait donc potentiellement passer la variable en Booléen  
        <br>
        - `contact`  
            - Moyen de contact avec le client  
            - Contient des valeurs manquantes  
        <br>
        - `month`  
            - Mois du dernier contact avec le client  
            - Pas de valeurs manquantes  
            - Il est écrit en texte, à voir s'il serait pertinent de regrouper le jour et le mois ensemble pour faire une analyse temporelle  
        <br>
        - `poutcome`  
            - Résultat de la campagne précédente  
            - Contient une majorité de valeurs manquantes "unknown"  
            - Ce qui en fait une donnée peu pertinente dans la suite des analyses  
        <br>
        - `deposit`  
            - C'est la valeur cible qui nous intéresse et qui nous dit si la campagne marketing a été un succès (le client a souscrit au compte à terme : yes) ou non (no)  
            - Pas de valeurs manquantes  
            - C'est donc cette variable qui sera principalement utilisée dans les analyses bivariées  
            - On voit qu'il n'y a pas de déséquilibre (52% vs 47 %) et donc potentiellement pas de besoin de rééchantillonnage
        """, unsafe_allow_html = True)


    with tab3:
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe().T)
        st.markdown("---")