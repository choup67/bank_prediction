import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import load_data
import textwrap
import inspect

def show_exploration():
    # Chargement des données
    df = load_data()

    # Préparation des données numériques
    def prepa_num(df) :
        balance_mediane = df['balance'].median()

        df['balance_group'] = df['balance'].apply(lambda x: '<0' if x < 0
                                                else f'0-{int(balance_mediane)}' if x <= balance_mediane
                                                else f'>{int(balance_mediane)}')

        df['age_group'] = pd.cut(df['age'], bins = [18, 32, 38, 48, 95], labels = ['18-32', '33-38', '39-48', '49-95'])

        balance_order = pd.CategoricalDtype(categories = ['<0', '0-550', '>550'], ordered = True)
        df['balance_group'] = df['balance_group'].astype(balance_order)
        return df

    # Préparation des données catégorielles
    def prepa_cat(df) : 
        df['month'] = pd.Categorical(df['month'],
                                    categories = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                                    ordered = True)

        job_mapping = {
            'management': 'Management',
            'entrepreneur': 'Independant', 'self-employed': 'Independant',
            'housemaid': 'Services', 'services': 'Services',
            'technician': 'Ouvriers-Techniciens', 'admin.': 'Ouvriers-Techniciens', 'blue-collar': 'Ouvriers-Techniciens',
            'retired': 'Autres', 'unemployed': 'Autres', 'student': 'Autres'
        }
        df['job_group'] = df['job'].map(job_mapping).fillna('Autres')
        return df


    st.title("Exploration et visualisation des données")
    st.subheader("Préparation des données")
    st.markdown("""
    Dans la découverte des données, on a pu constater que la variable `job` a beaucoup de modalités. On va donc réduire le nombre
                de modalités de cette variable en regroupant les modalités peu fréquentes dans une catégorie `autres`.
                Et faire également des regroupements par type de métier.""")
    
    st.code(textwrap.dedent(inspect.getsource(prepa_cat)), language = 'python')
    
    st.markdown("""
    Il en va de même pour la variable `age`que l'on va regrouper par tranche d'âge. <br>
    Enfin, pour la `balance`, on va également créer des sous groupes pour éviter les valeurs extrêmes.
    """, unsafe_allow_html = True)

    st.code(textwrap.dedent(inspect.getsource(prepa_num)), language = 'python')
    # Affichage variables numériques vs deposit
    st.subheader("Comparaison des variables numériques avec la variable cible")


    df = prepa_num(df)

    age_percentages = pd.crosstab(df['age_group'], df['deposit'], normalize = 'index') * 100
    balance_percentages = pd.crosstab(df['balance_group'], df['deposit'], normalize = 'index') * 100
    day_percentages = pd.crosstab(df['day'], df['deposit'], normalize = 'index') * 100

    fig, axes = plt.subplots(1, 3, figsize = (18, 5))

    axes[0].bar(age_percentages.index, age_percentages['no'], color = '#A1C9F4', label = 'no')
    axes[0].bar(age_percentages.index, age_percentages['yes'], bottom = age_percentages['no'],
                color = '#FFB482', label = 'yes')
    axes[0].set_title("Âge vs Deposit")
    axes[0].set_xlabel("Tranche d'âge")
    axes[0].set_ylabel("Pourcentage")
    axes[0].legend(title = 'Deposit')
    axes[0].grid(axis = 'y', linestyle = '--', alpha = 0.3)

    axes[1].bar(balance_percentages.index, balance_percentages['no'], color = '#A1C9F4', label = 'no')
    axes[1].bar(balance_percentages.index, balance_percentages['yes'], bottom = balance_percentages['no'],
                color = '#FFB482', label = 'yes')
    axes[1].set_title("Solde vs Deposit")
    axes[1].set_xlabel("Groupe de solde")
    axes[1].legend(title = 'Deposit')
    axes[1].grid(axis = 'y', linestyle = '--', alpha = 0.3)

    axes[2].bar(day_percentages.index, day_percentages['no'], color = '#A1C9F4', label = 'no')
    axes[2].bar(day_percentages.index, day_percentages['yes'], bottom = day_percentages['no'],
                color = '#FFB482', label = 'yes')
    axes[2].set_title("Jour du mois vs Deposit")
    axes[2].set_xlabel("Jour du mois")
    axes[2].legend(title = 'Deposit')
    axes[2].grid(axis = 'y', linestyle = '--', alpha = 0.3)
    axes[2].set_xticks(range(1, 32))
    axes[2].set_xticklabels(range(1, 32), fontsize = 8)

    st.pyplot(fig)



    # Affichage des variables catégorielles vs deposit
    pastel_colors = ["#A1C9F4", "#FFB482"]

    groupes = {
        "Données démographiques": ['job_group', 'marital', 'education'],
        "Données financières": ['default', 'housing', 'loan'],
        "Données campagne": ['contact', 'month', 'poutcome']} 
    
    df = prepa_cat(df)

    st.subheader("Comparaison des variables catégorielles avec la variable cible")

    for groupe, variables in groupes.items():
        with st.expander(f"{groupe}"):
            fig, axes = plt.subplots(nrows = 1, ncols = len(variables), figsize = (5 * len(variables), 5))
            fig.suptitle(f"{groupe}", fontsize = 16, fontweight = 'bold')

            # Si une seule variable, axes ne sera pas une liste => forcer en liste
            if len(variables) == 1:
                axes = [axes]

            for i, var in enumerate(variables):
                cross_tab = pd.crosstab(df[var], df['deposit'], normalize = 'index') * 100
                cross_tab.plot(kind = 'bar', ax = axes[i], color = pastel_colors)
                axes[i].set_title(f"{var} vs deposit")
                axes[i].set_ylabel("Pourcentage")
                axes[i].set_xlabel(var)
                axes[i].legend(title = "Deposit")
                axes[i].tick_params(axis = 'x', rotation = 90)

            plt.tight_layout(rect = [0, 0, 1, 0.92])
            st.pyplot(fig)