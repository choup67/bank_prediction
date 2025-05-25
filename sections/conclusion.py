import streamlit as st

def show_conclusion():
    st.title("Conclusions")
    tab1, tab2 = st.tabs(["Observations", "Améliorations continues"])
    with tab1:
        st.subheader("Résultats observés, conclusions principales")
        st.markdown("""
        - On peut constater qu'en travaillant nos variables explicatives, on peut améliorer les performances de nos différents modèles
        - Certaines variables explicatives ont peu d'impact sur les résultats et peuvent donc être supprimées
        - Le paramétrage des modèles a un grand impact sur les résultats. Trouver les bons paramètres permettant d'obtenir le résultat souhaité est un challenge
        - En fonction des attentes métiers on pourra au choix :
            - Privilégier le modèle Random Forest v2 (`n_estimators` = 100, `max_depth` = 10, `min_samples_leaf` = 15) si l'objectif est de réduire les faux positifs et faux négatifs en ayant un bon équilibre (f1 score)
            - Privilégier le modèle Gradient Boosting (`n_estimators` = 100, `learning_rate` = 0.05, `max_depth` = 5, `min_samples_split` = 20, `min_samples_leaf` = 15), si la priorité est d'éliminer les faux positifs
                    """, unsafe_allow_html=True)
    with tab2:
        st.subheader("Pistes d'améliorations continues")
        st.markdown("""
        - Il peut être intéressant de continuer à explorer les paramètres des modèles mais en utilisant des fonctions qui permettent de tester et obtenir les meilleurs paramètres (GridSearchCV(), RandomizedSearchCV(), Optuna, Hyperopt, ...)
        - Il peut aussi être intéressant de faire une cross-validation pour éliminer le fait que le découpage appliqué ait pu avoir un impact dans un sens ou l'autre
        - On pourra également continuer à travailler sur les variables explicatives en cherchant à créer de nouvelles variables ou en transformant celles existantes pour améliorer les performances des modèles
        - Enfin on pourra également réflechir à l'implémentation de modèles plus complexes comme les réseaux de neurones ou les modèles de type XGBoost, LightGBM, CatBoost, etc.
                    """)