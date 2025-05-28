import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

        st.subheader("Remarques générales")
        if st.checkbox("Afficher les remarques générales sur les variables catégorielles", value = False):
            st.markdown("""
        - On remarque que les variables ont le plus souvent entre 2 et 4 modalités. Sauf pour la variable `job` et évidemment les mois.
        - La variable `job` contient 12 modalités différentes, ce qui est relativement élevé. On pourrait donc envisager de regrouper certaines modalités pour simplifier l'analyse.
        
        - On peut aussi noter l’apparition de la valeur `unknown` pour certaines variables : `job`, `education`, `contact` et `poutcome`.  Ce qui laisse sous-entendre que même si on n’avait pas de valeurs nulles dans le dataset, il y a tout de même des données manquantes pour certaines catégories.  
                    
        - Dans certains cas, l’impact est faible car peu de données manquantes, mais pour d’autres variables comme `poutcome`, la majorité des données sont `unknown`.  
          Nous éliminerons donc les variables qui ne sont pas pertinentes pour l’analyse.

        - Certaines variables contiennent les valeurs `no` et `yes`. Comme ces variables ne contiennent pas de valeurs manquantes, nous les tranformons en type booléen. Cela permettra d’optimiser les calculs, la mémoire et la manipulation des données.  
        """, unsafe_allow_html = True)

        st.subheader("Détail des variables catégorielles")
        if st.checkbox("Afficher les observations sur les variables catégorielles", value = False):
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
            - C'est la valeur cible qui nous intéresse et qui nous dit si la campagne marketing a été un succès (le client a souscrit au compte à terme : yes ou no)  
            - Pas de valeurs manquantes  
            - C'est donc cette variable qui sera principalement utilisée dans les analyses bivariées  
            - On voit qu'il n'y a pas de déséquilibre (52% vs 47 %) et donc potentiellement pas de besoin de rééchantillonnage
        """, unsafe_allow_html = True)


    with tab3:
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe().T)
        st.markdown("---")

        # Stockage des variables numériques 
        var_num = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

        st.subheader("Analyse des outliers (valeurs extrêmes)")
        # Calcul des outliers et stockage des résultats
        outlier_data = []

        for col in var_num:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            limite_inf = Q1 - 1.5 * IQR
            limite_sup = Q3 + 1.5 * IQR
            outliers = df[(df[col] < limite_inf) | (df[col] > limite_sup)][col]
            pourcentage = len(outliers) / len(df) * 100
            outlier_data.append({
                'Variable': col,
                'Pourcentage d\'outliers': round(pourcentage, 2)
            })

        # Création du DataFrame des résultats
        df_outliers = pd.DataFrame(outlier_data)

        # Affichage du df
        st.dataframe(df_outliers)


        st.subheader("Affichage de la distribution des variables numériques")
        # Création de la figure
        fig, axs = plt.subplots(2, 4, figsize = (16, 8))  # 2 lignes, 4 colonnes

        # Création des boxplots
        for i, col in enumerate(var_num):
            ax = axs[i // 4, i % 4]
            sns.boxplot(x = 'deposit', y = col, data = df, ax = ax)
            ax.set_title(f'Distribution de {col} selon deposit')
            ax.tick_params(axis = 'x', rotation = 45)

        plt.tight_layout()
        # Affichage dans Streamlit
        st.pyplot(fig)


        st.subheader("Remarques générales")
        if st.checkbox("Afficher les remarques générales sur les variables numériques", value = False):
            st.markdown("""
        - On peut noter des valeurs négatives pour certaines variables. Dans le cas du solde par exemple, cela est cohérent
            - Pour les autres variables, il semble que -1 signifie que les données sont manquantes.
        - L'écart type de certaines variables est très élevé. Cela indique une grande disperssion et une moyenne potentiellement tirée vers le haut par des valeurs extrêmes. 
        """, unsafe_allow_html = True)


        st.subheader("Détail des variables numériques")
        if st.checkbox("Afficher les observations sur les variables numériques", value = False):
            st.markdown("""
        - `age`
            - **Age du client**
            - La moyenne est de 41 ans
            - Le client le plus jeune a 18 ans
            - Le client le plus âgé à 95 ans
            - 50% des clients ont entre 32 et 39 ans
        - `balance`
            - **Solde du compte**
            - Il y a une grande disperssion des données avec un min négatif et un max très important
            - 75% des clients ont moins de 1708 dollars sur leurs comptes
            - Le max a 81 204 dollars impacte la moyenne qui est 3 fois supérieure à la médiane
        - `day`
            - **jour du dernier contact**
        - `duration`
            - **Durée du dernier contact en seconde**
            - La moyenne d'un contact est de 372 secondes, soit environ 6 minutes
            - 50% des appels durent entre 138 et 496 secondes (2 à 8 minutes)
            - On voit là aussi des valeurs très importantes avec un max à 3881 secondes soit un appel de plus d'une heure
        - `campaign`
            - **Nombre de contact avec le client durant la campagne**
            - En moyenne les clients sont contactés 2,5 fois
            - 75% des clients sont contactés 3 fois ou moins
            - On sait donc rapidement si un client va décider de souscrire ou non
        - `pdays`
            - **Nombre de jours depuis le dernier contact avec le client depuis la précédente campagne**
            - La valeur -1 indique que le client n'a pas été contacté précédemment ou que la valeur est manquante
            - Plus de 50% des clients n'ont pas été contactés lors d'une précédente campagne
        - `previous`
            - **Nombre de contact précédent la campagne en cours**
            - La valeur 0 indique que le client n'a pas été contacté précédemment
            - 75% des clients ont été contactés une fois ou moins
            - On peut déduire que la nouvelle campagne cherche à toucher de nouveaux prospects
            - On a aussi des valeurs extrêmes où un client a été contacté 58 fois
            - Peut être que le groupe de client déjà contacté a toujours souscrit aux offres précédentes et serait donc une bonne cible stratégique pour la nouvelle campagne. Chose que l'on pourra vérifier lors des analyses graphiques
        """, unsafe_allow_html = True)