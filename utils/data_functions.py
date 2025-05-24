import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)

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

# Suppression des colonnes inutiles
@st.cache_data
def remove_useless_col(df): 
    df = df.drop(['poutcome', 'duration', 'pdays', 'campaign', 'contact', 'age', 'job', 'balance', 'loan', 'housing'], axis=1)
    return df

# Sépration du jeu de données train/test et définition des variables explicatives et cibles
@st.cache_data
def split_data(df):
    # Séparer les variables cibles et les variables explicatives
    y = df['deposit']
    X = df.drop('deposit', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

    # Définir les colonnes numériques et catégorielles
    var_num = ['day', 'default', 'active_loan', 'previous']
    var_cat = ['job_group', 'marital', 'education', 'month','age_group', 'balance_group']

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


@st.cache_data
def ready_to_process_data():
    from utils.data_loader import load_cleaned_and_prepared_data 
    df = load_cleaned_and_prepared_data()
    # Séparer les données en train et test
    df, X_train, X_test, y_train, y_test, var_num, var_cat = split_data(df)
    # Gérer les valeurs manquantes
    X_train, X_test = manage_missing_values(X_train, X_test, var_num, var_cat)
    # Mettre à l'échelle les variables numériques
    X_train, X_test = scale_numeric_features(X_train, X_test, var_num)
    # Encoder les variables catégorielles
    X_train_final, X_test_final, y_train, y_test = encode_categorical_features(
    X_train, X_test, y_train, y_test, var_cat)
    return X_train_final, X_test_final, y_train, y_test

@st.cache_data
def evaluate_models():
    # Chargement des données prétraitées
    X_train, X_test, y_train, y_test = ready_to_process_data()
    
    # Modèles à tester
    models = {
        "Logistic Regression": LogisticRegression(max_iter = 1000),
        "Random Forest": RandomForestClassifier(n_estimators = 100, random_state = 48),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(random_state = 48)
    }

    results_list = []
    reports = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision_1 = precision_score(y_test, y_test_pred, pos_label = 1)
        recall_1 = recall_score(y_test, y_test_pred, pos_label = 1)
        f1_1 = f1_score(y_test, y_test_pred, pos_label = 1)

        cm = confusion_matrix(y_test, y_test_pred)
        false_positives = cm[0][1]

        results_list.append({
            "Modèle": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Faux positifs": false_positives,
            "Precision (classe 1)": round(precision_1, 4),
            "Recall (classe 1)": round(recall_1, 4),
            "F1-score (classe 1)": round(f1_1, 4)
        })

        reports[name] = classification_report(y_test, y_test_pred, output_dict = False)
        confusion_matrices[name] = cm

    return pd.DataFrame(results_list), reports, confusion_matrices, models

@st.cache_data
def evaluate_models_optimisation():
    # Chargement des données prétraitées
    X_train, X_test, y_train, y_test = ready_to_process_data()

    # Modèles à tester
    models = {
        "Logistic Regression": LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=48),
        "SVM": SVC(random_state=48),
        "Random Forest v2": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=15, random_state=48),
        "Decision Tree v2": DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=48),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, min_samples_split=20,
        min_samples_leaf=15, random_state=48)
    }

    results_list = []
    reports = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision_1 = precision_score(y_test, y_test_pred, pos_label = 1)
        recall_1 = recall_score(y_test, y_test_pred, pos_label = 1)
        f1_1 = f1_score(y_test, y_test_pred, pos_label = 1)

        cm = confusion_matrix(y_test, y_test_pred)
        false_positives = cm[0][1]

        results_list.append({
            "Modèle": name,
            "Train Accuracy": round(train_acc, 4),
            "Test Accuracy": round(test_acc, 4),
            "Faux positifs": false_positives,
            "Precision (classe 1)": round(precision_1, 4),
            "Recall (classe 1)": round(recall_1, 4),
            "F1-score (classe 1)": round(f1_1, 4)
        })

        reports[name] = classification_report(y_test, y_test_pred, output_dict = False)
        confusion_matrices[name] = cm

    return pd.DataFrame(results_list), reports, confusion_matrices, models



def importance_features():
    # Récupération des modèles entraînés
    _, _, _, trained_models = evaluate_models_optimisation()

    # Récupération du modèle Gradient Boosting
    model = trained_models.get("Gradient Boosting")

    if model is None or not hasattr(model, "feature_importances_"):
        st.warning("Le modèle Gradient Boosting n'est pas disponible ou ne fournit pas d'importances.")
        return

    # Chargement des données pour récupérer les noms des colonnes
    X_train, _, _, _ = ready_to_process_data()
    feature_names = X_train.columns
    importances = model.feature_importances_

    # Créer le DataFrame
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by = "Importance", ascending = False)

    # Affichage
    plt.barh(feature_importance_df["Feature"][:10], feature_importance_df["Importance"][:10], color = 'skyblue')
    plt.xlabel("Importance")
    plt.title("Top 10 Features - Gradient Boosting")
    plt.gca().invert_yaxis()
    st.pyplot(plt)