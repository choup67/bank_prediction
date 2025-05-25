import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Suppression des colonnes inutiles
@st.cache_data
def remove_useless_col(df): 
    df = df.drop(['poutcome', 'duration', 'pdays', 'campaign', 'contact', 'age', 'job', 'balance', 'loan', 'housing'], axis=1)
    return df

@st.cache_data
def remove_useless_col_advanced(df): 
    df = df.drop(['poutcome', 'duration', 'pdays', 'campaign', 'age', 'job', 'balance', 'contact', 'month', 'previous'], axis=1)
    return df


# --- Fonctions de préparation des données pour la modélisation ---
# Sépration du jeu de données train/test et définition des variables explicatives et cibles
@st.cache_data
def split_data(df):
    y = df['deposit']
    X = df.drop('deposit', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

    # Définir les colonnes numériques et catégorielles
    var_num = ['day', 'default', 'active_loan', 'previous']
    var_cat = ['job_group', 'marital', 'education', 'month','age_group', 'balance_group']

    return df, X_train, X_test, y_train, y_test, var_num, var_cat


@st.cache_data
def split_data_advanced(df):
    y = df['deposit']
    X = df.drop('deposit', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)
    
    # Encodage cyclique pour day
    X_train['day_sin'] = np.sin(2 * np.pi * X_train['day'] / 31)
    X_train['day_cos'] = np.cos(2 * np.pi * X_train['day'] / 31)
    X_test['day_sin'] = np.sin(2 * np.pi * X_test['day'] / 31)
    X_test['day_cos'] = np.cos(2 * np.pi * X_test['day'] / 31)

    # Encodage cyclique pour month_num
    X_train['month_sin'] = np.sin(2 * np.pi * X_train['month_num'] / 12)
    X_train['month_cos'] = np.cos(2 * np.pi * X_train['month_num'] / 12)
    X_test['month_sin'] = np.sin(2 * np.pi * X_test['month_num'] / 12)
    X_test['month_cos'] = np.cos(2 * np.pi * X_test['month_num'] / 12)

    # On supprime les colonnes originales
    X_train = X_train.drop(['day', 'month_num'], axis=1)
    X_test = X_test.drop(['day', 'month_num'], axis=1)

    # Définir les colonnes numériques et catégorielles
    var_num = ['default', 'active_loan', 'day_sin', 'day_cos', 'month_sin', 'month_cos', 'contacted_before']
    var_cat = ['job_group', 'marital', 'education', 'age_group', 'balance_group']

    return df, X_train, X_test, y_train, y_test, var_num, var_cat

# Gestion des valeurs manquantes dans les données
@st.cache_data
def manage_missing_values(X_train, X_test, var_num, var_cat):
    # Instancier les imputateurs
    num_imputer = SimpleImputer(strategy='median')  # Médiane pour les colonnes numériques
    cat_imputer = SimpleImputer(strategy='most_frequent')  # Mode pour les colonnes catégorielles

    # Appliquer les imputateurs aux colonnes numériques
    X_train[var_num] = num_imputer.fit_transform(X_train[var_num])
    X_test[var_num] = num_imputer.transform(X_test[var_num])

    # Appliquer les imputateurs aux colonnes catégorielles
    X_train[var_cat] = cat_imputer.fit_transform(X_train[var_cat])
    X_test[var_cat] = cat_imputer.transform(X_test[var_cat])

    return X_train, X_test

# Mise à l'échelle des valeurs numériques
@st.cache_data
def scale_numeric_features(X_train, X_test, var_num):
    sc = StandardScaler()
    X_train[var_num] = sc.fit_transform(X_train[var_num])
    X_test[var_num] = sc.transform(X_test[var_num])

    return X_train, X_test

# Encodage des variables catégorielles
@st.cache_data
def encode_categorical_features(X_train, X_test, y_train, y_test, var_cat):
    # Réinitialiser les index
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)


    # Initialiser l'encodeur OneHotEncoder
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    # Entraîner l'encodeur sur le jeu d'entraînement
    ohe.fit(X_train[var_cat])

    # Appliquer la transformation sur les jeux d'entraînement et de test
    X_train_encoded = ohe.transform(X_train[var_cat])
    X_test_encoded = ohe.transform(X_test[var_cat])

    # Convertir en DataFrame en conservant les noms de colonnes et les index
    X_train_encoded_df = pd.DataFrame(X_train_encoded,
                                    columns=ohe.get_feature_names_out(var_cat),
                                    index=X_train.index)

    X_test_encoded_df = pd.DataFrame(X_test_encoded,
                                    columns=ohe.get_feature_names_out(var_cat),
                                    index=X_test.index)

    # Concaténer avec les variables numériques restantes
    X_train_final = pd.concat([X_train.drop(columns=var_cat), X_train_encoded_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=var_cat), X_test_encoded_df], axis=1)
    
    return X_train_final, X_test_final, y_train, y_test