import streamlit as st
from utils.data_functions import evaluate_models, evaluate_models_optimisation, importance_features
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from io import BytesIO

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

            # Afficher sans étirement
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
        results_df_optim, _, _, _ = evaluate_models_optimisation()
        st.dataframe(results_df_optim, use_container_width = True)
        st.markdown("""
        On peut tout de suite constater que les résultats sont meilleurs, 
        et que nous avons réduit l'overfitting des modèles Decision Tree et Random Forest.
        Le nouveau modele Gradient Boosting testé semble également très prometteur.
                    """)
        
        st.subheader("Affichage des importances features sur Gradient Boosting")
        if st.checkbox("Afficher les features importantes", value = False):
            importance_features()
    with tab2:
        st.write("Optimisation par grid search, random search, etc.")