import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# --- Fonctions de préparation des données ---
 # Préparation des données numériques
@st.cache_data
def prepa_num(df) :
    balance_mediane = df['balance'].median()

    df['balance_group'] = df['balance'].apply(lambda x: '<0' if x < 0
                                            else f'0-{int(balance_mediane)}' if x <= balance_mediane
                                            else f'>{int(balance_mediane)}')

    df['age_group'] = pd.cut(df['age'], bins = [18, 32, 38, 48, 95], labels = ['18-32', '33-38', '39-48', '49-95'])

    balance_order = pd.CategoricalDtype(categories = ['<0', '0-550', '>550'], ordered = True)
    df['balance_group'] = df['balance_group'].astype(balance_order)
    return df

# préparation des données catégorielles
@st.cache_data
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

# Remplacement des valeurs "unknown" par NaN
@st.cache_data
def replace_unknown(df):
    df.replace("unknown", np.nan, inplace = True)
    return df # type: ignore

# Transformation des variables à deux modalités en booléennes
@st.cache_data
def transform_to_bool(df):
    df[["housing", "default", "loan", "deposit"]] = df[["housing", "default", "loan", "deposit"]].replace({"yes": 1, "no": 0})
    return df

# Création de la variable active_loan
@st.cache_data
def create_active_loan(df):
    df['active_loan'] = (df['housing'] | df['loan']).astype(int)
    return df

# Création de transofrmations avancées (mois clyclique, contacted_before)

@st.cache_data
def advanced_transformations(df):
    # Mapping des mois texte vers numériques
    months_mapping = {
        'jan': 1, 'feb': 2, 'mar': 3,
        'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12
    }

    # Conversion de la colonne 'month' en string pour éviter les conflits avec Categorical
    df['month_num'] = df['month'].astype(str).map(months_mapping).fillna(0).astype(int)

    # Création de la variable binaire contacted_before à partir de previous
    df['contacted_before'] = df['previous'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)

    return df