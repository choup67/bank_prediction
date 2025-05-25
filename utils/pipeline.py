from utils.transformations import prepa_num, prepa_cat, replace_unknown, create_active_loan, transform_to_bool, advanced_transformations
from utils.data_functions import remove_useless_col, split_data_advanced, remove_useless_col_advanced, split_data, manage_missing_values, scale_numeric_features, encode_categorical_features
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from utils.data_loader import load_data
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# On charge les données et on applique les transformations sur les variables catégorielles et numériques
@st.cache_data
def load_and_prepare_data():
    df = load_data()
    df = prepa_num(df)
    df = prepa_cat(df)
    return df

# -----------------

# On gère les valeurs manquantes et on enrichit les données
@st.cache_data
def load_cleaned_and_prepared_data():
    df = load_and_prepare_data()
    df = replace_unknown(df)
    df = transform_to_bool(df)
    df = create_active_loan(df)
    return df

# Application de transformations avancées
@st.cache_data
def load_cleaned_and_prepared_data_advanced():
    df = load_cleaned_and_prepared_data()
    df = advanced_transformations(df)
    return df

# -------------------

# Premières itérations modélisation

# Préparation des données pour la modélisation
@st.cache_data
def ready_to_process_data():
    # On charge les données préparées puis on supprime les colonnes inutiles
    df = load_cleaned_and_prepared_data()
    df = remove_useless_col(df)

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

    # Evaluation des modèles
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

# -------------------
# Optimisation 
# Préparation des données pour la modélisation optimisée
@st.cache_data
def ready_to_process_data_advanced():
    # On charge les données préparées puis on supprime les colonnes inutiles
    df = load_cleaned_and_prepared_data_advanced()
    df = remove_useless_col_advanced(df)

    # Séparer les données en train et test
    df, X_train, X_test, y_train, y_test, var_num, var_cat = split_data_advanced(df)
    # Gérer les valeurs manquantes
    X_train, X_test = manage_missing_values(X_train, X_test, var_num, var_cat)
    # Mettre à l'échelle les variables numériques
    X_train, X_test = scale_numeric_features(X_train, X_test, var_num)
    # Encoder les variables catégorielles
    X_train_final, X_test_final, y_train, y_test = encode_categorical_features(
    X_train, X_test, y_train, y_test, var_cat)
    return X_train_final, X_test_final, y_train, y_test



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