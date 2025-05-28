import streamlit as st
import pandas as pd
from utils.pipeline import evaluate_models, evaluate_models_optimisation, importance_features
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.pipeline import ready_to_process_data_advanced

def show_modelisation():

    st.title("Modélisation")
    tab1, tab2 = st.tabs(["Premières itérations et interprétation", "Optimisations"])
    
    with tab1:
        st.subheader("Évaluation des modèles et performances")

        # Évaluation
        results_df, reports, matrices, models = evaluate_models()

        # Tableau récapitulatif
        st.dataframe(results_df, use_container_width = True)

        st.markdown("""
        - Il y a beaucoup d'overfitting sur le random forest et le decision tree
        - On remarque que le modèle prédit mieux les clients qui ne soucriront pas (classe 0) que ceux qui souscriront
        - Il est possible de se focaliser sur la réduction des faux positifs (afin de ne pas contacter inutilement des gens qui ne souscriront pas) mais il semble également pertinent de réduire les faux négatifs (afin de ne pas oublier de contacter des gens succeptible de souscrire). C'est pour ça qu'il faudra également prendre en compte le f1-Score qui est l'équilibre entre les deux
        - A ce stade de l'étude, le modèle SVC semble le plus performant avec très peu de faux positifs et un bon score F1
        """, unsafe_allow_html = True)

        # Détails par modèle
        st.subheader("Détails par modèle")
        if st.checkbox("Afficher le détail par modèle", value = False):

            selected_model = st.selectbox("Choisir un modèle", options = list(models.keys()))

            # Affichage des résultats du modèle sélectionné
            st.markdown(f"**Rapport de classification – {selected_model}**")
            st.text(reports[selected_model])

            # Matrice de confusion
            fig, ax = plt.subplots(figsize = (4, 3))  # taille réelle du canvas
            disp = ConfusionMatrixDisplay(
                confusion_matrix = matrices[selected_model],
                display_labels = models[selected_model].classes_
            )
            disp.plot(ax = ax, cmap = plt.cm.Blues, colorbar = False)
            plt.tight_layout()

            # Enregistrer en mémoire
            buf = BytesIO()
            fig.savefig(buf, format = "png", dpi = 100, bbox_inches = "tight")
            buf.seek(0)
            plt.close(fig)

            # Affichage retravaillé
            st.image(buf, caption = f"Matrice de confusion – {selected_model}", use_container_width = False)

        st.subheader("Interprétation des résultats")
        st.markdown("""
        Suite aux conclusions des différentes approches, nous avons décidé d'appliquer les optimisations suivantes :
        - Modification des paramètres de profondeur pour éviter que le modèle surapprend
        - Ajustement du nombre d'échantillons minimum par feuille pour rendre le modèle moins complexe
            - DecisionTree : max_depth=10, min_samples_leaf=10
            - RandomForrest : n_estimators=100, max_depth=15, min_samples_leaf=5
        - Ajout de modèles supplémentaires
                    """)
        
        st.subheader("Résultats après premiers ajustements")
        # Affichage des résultats après optimisation
        results_df_optim, _, _, _ = evaluate_models_optimisation()
        st.dataframe(results_df_optim, use_container_width = True)
        st.markdown("""
        On peut tout de suite constater que les résultats sont meilleurs, 
        et que nous avons réduit l'overfitting des modèles Decision Tree et Random Forest.
        Le nouveau modele Gradient Boosting testé semble également très prometteur.
                    """)
        
        # Importance features sur le meilleur modèle
        st.subheader("Affichage des importances features sur Gradient Boosting")
        if st.checkbox("Afficher les features importantes", value = False):
            importance_features()


    with tab2:

        # Création d'un outil pour tester manuellement plusieurs configurations possibles
        st.header("Modélisation personnalisée")
        colA, colB = st.columns(2)

        with colA:
            st.subheader("Conclusions et modifications")
            st.markdown("""
            Dans cette section, nous allons affiner notre modèle en appliquant les optimisations suivantes :
            - Suppression de variables à faible impact (`contact`)  
            - Encodage cyclique pour les `mois` et `jours`  
            - Transformation de `previous` en `contacted_before` (booléen)  
            """)

        with colB:
            st.subheader("Présentation de l'outil")
            st.markdown("""
            Vous pouvez utiliser l'outil ci-dessous pour :
            - Choisir un modèle parmi une liste  
            - Sélectionner les paramètres du modèle  
            - Entraîner le modèle  
            - Afficher les résultats (accuracy, precision, recall, f1-score)  
            - Sauvegarder les résultats dans un tableau récapitulatif  
            - Réinitialiser le tableau récapitulatif  
            """)

        st.markdown("---")
        st.subheader("Entraînement de modèles personnalisés")
        # Initialisation
        if "results_df" not in st.session_state:
            st.session_state["results_df"] = pd.DataFrame(columns = ["Modèle", "Paramètres", "Accuracy", "Precision", "Recall", "F1"])
        for metric in ["acc", "prec", "rec", "f1"]:
            if metric not in st.session_state:
                st.session_state[metric] = None

        # Chargement des données
        X_train, X_test, y_train, y_test = ready_to_process_data_advanced()

        # Modèle et paramètres par défaut
        model_options = ["Logistic Regression", "SVM", "Random Forest v2", "Decision Tree v2", "Gradient Boosting"]
        model_choice = st.selectbox("Choisissez un modèle :", model_options)

        params = {}

        param_col1, param_col2 = st.columns(2)

        if model_choice == "Logistic Regression":
            with param_col1:
                C = st.selectbox("C (Regularization)", [0.01, 0.1, 1, 10], index = 2)
            with param_col2:
                class_weight = st.selectbox("Class Weight", ["balanced", None])
            params = {"C": C, "class_weight": class_weight, "max_iter": 1000, "random_state": 48}
            model = LogisticRegression(**params)

        elif model_choice == "SVM":
            with param_col1:
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            with param_col2:
                C = st.selectbox("C (Regularization)", [0.1, 1, 10])
            params = {"kernel": kernel, "C": C, "random_state": 48}
            model = SVC(**params)

        elif model_choice == "Random Forest v2":
            with param_col1:
                n_estimators = st.selectbox("n_estimators", [50, 100, 200], index = 1)
                max_depth = st.selectbox("max_depth", [5, 10, 15])
            with param_col2:
                min_samples_leaf = st.selectbox("min_samples_leaf", [5, 10, 15])
            params = {
                "n_estimators": n_estimators, "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf, "random_state": 48
            }
            model = RandomForestClassifier(**params)

        elif model_choice == "Decision Tree v2":
            with param_col1:
                max_depth = st.selectbox("max_depth", [5, 10, 15])
            with param_col2:
                min_samples_leaf = st.selectbox("min_samples_leaf", [5, 10, 15])
            params = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "random_state": 48}
            model = DecisionTreeClassifier(**params)

        elif model_choice == "Gradient Boosting":
            with param_col1:
                n_estimators = st.selectbox("n_estimators", [50, 100, 200], index = 1)
                learning_rate = st.selectbox("learning_rate", [0.01, 0.05, 0.1], index = 1)
            with param_col2:
                max_depth = st.selectbox("max_depth", [3, 5, 7])
                min_samples_split = st.selectbox("min_samples_split", [10, 20, 30])
                min_samples_leaf = st.selectbox("min_samples_leaf", [5, 10, 15])
            params = {
                "n_estimators": n_estimators, "learning_rate": learning_rate,
                "max_depth": max_depth, "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf, "random_state": 48
            }
            model = GradientBoostingClassifier(**params)

        # Entraînement
        if st.button("Entraîner le modèle"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.session_state["acc"] = accuracy_score(y_test, y_pred)
            st.session_state["prec"] = precision_score(y_test, y_pred, pos_label = 1)
            st.session_state["rec"] = recall_score(y_test, y_pred, pos_label = 1)
            st.session_state["f1"] = f1_score(y_test, y_pred, pos_label = 1)

        st.markdown("---")

        # Affichage des résultats
        if st.session_state["acc"] is not None:
            st.subheader("Résultats")
            st.write(f"**Accuracy :** {st.session_state['acc']:.4f}")
            st.write(f"**Precision (classe 1) :** {st.session_state['prec']:.4f}")
            st.write(f"**Recall (classe 1) :** {st.session_state['rec']:.4f}")
            st.write(f"**F1-score (classe 1) :** {st.session_state['f1']:.4f}")

        # Bouton sauvegarder pour mettre les résultats dans un dataframe
        if st.button("💾 Sauvegarder les résultats"):
            if st.session_state["acc"] is not None:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                new_result = {
                    "Modèle": model_choice,
                    "Paramètres": param_str,
                    "Accuracy": st.session_state["acc"],
                    "Precision": st.session_state["prec"],
                    "Recall": st.session_state["rec"],
                    "F1": st.session_state["f1"]
                }
                st.session_state["results_df"] = pd.concat(
                    [st.session_state["results_df"], pd.DataFrame([new_result])],
                    ignore_index = True
                )

        st.markdown("---")

        # Affichage du dataframe avec les résultats sauvegardés
        st.subheader("Tableau récapitulatif des modèles sauvegardés")
        st.dataframe(st.session_state["results_df"])

        # Affichage du bouton effacer pour vider le dataframe
        if st.button("🧹 Effacer la table des résultats"):
            st.session_state["results_df"] = pd.DataFrame(columns = ["Modèle", "Paramètres", "Accuracy", "Precision", "Recall", "F1"])
            st.success("Tableau réinitialisé.")