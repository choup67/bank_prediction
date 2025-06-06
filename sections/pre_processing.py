import streamlit as st
from utils.pipeline import load_and_prepare_data, ready_to_process_data
from utils.transformations import replace_unknown, create_active_loan, transform_to_bool
from utils.data_functions import remove_useless_col
import textwrap
import inspect


def show_pre_processing():
    # Chargement des données préparées
    df = load_and_prepare_data()

    st.title("Nettoyage et Pré processing")
    tab1, tab2 = st.tabs(["Nettoyage", "Pré Processing"])
    with tab1:
        st.subheader("Récupération des données enrichies")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("Remplacement des 'unknown' par des NaN")
        st.markdown("""
        On a vu que certaines variables contiennent des valeurs `unknown` qui ne sont pas pertinentes pour l'analyse.
        On va donc les remplacer par des valeurs nulles (NaN) pour pouvoir les traiter plus facilement.
        """)

        # Remplacement des "unknown" par des NaN
        df = replace_unknown(df)
        st.code(textwrap.dedent(inspect.getsource(replace_unknown)), language = 'python')

        # Affichage des valeurs nulles par colonne
        if st.checkbox("Afficher le résultat", value = False):
            st.write(df.isnull().sum())

        st.subheader("Création de booleens pour les variables à deux modalités")
        st.markdown("""
        Comme vu dans la partie exploration, certaines variables n'ont que deux modalités (yes/no).
        On va donc les transformer en variables booléennes (0/1) pour faciliter l'analyse et améliorer les performances.
        """)

        # Conversion des variables "yes" et "no" en 1 et 0
        df = transform_to_bool(df)
        st.code(textwrap.dedent(inspect.getsource(transform_to_bool)), language = 'python')
        # affichage des variables converties
        if st.checkbox("Afficher les variables converties", value = False):
            st.write(df[["housing", "default", "loan", "deposit"]].head(5))

        # Création de la variable `active_loan` pour savoir si le client a au moins un prêt actif
        st.subheader("Création de la variable `Active Loan`")
        st.markdown("""
        On va créer une nouvelle variable `active_loan` qui vaudra 1 si le client a au moins un prêt actif (housing ou loan).
       """)
        df = create_active_loan(df)
        st.code(textwrap.dedent(inspect.getsource(create_active_loan)), language = 'python')
        # affichage de la nouvelle colonne
        if st.checkbox("Afficher la nouvelle colonne", value = False):
            st.write(df[['housing', 'loan', 'active_loan']].head(5))

        # Suppression des colonnes inutiles
        st.subheader("Suppression des colonnes inutiles")
        st.markdown("""
        - `contact` : contient des valeurs manquantes et n’est pas pertinent pour l’analyse
        - `poutcome` : majorité de valeurs manquantes, inutilisable
        - `duration` : on ne connait pas la valeur en avance, donc elle ne peut pas être utilisée comme variable prédictive
        - `pdays` : plus d’1/4 d’outliers, il n’est pas pertinent de garder cette variable
        - `campaign` : indique le nombre de fois que le client a déjà été contacté durant la campagne comme pour `duration`, elle ne peut pas être utilisée comme variable prédictive
        -  `age, job et balance, loan et housing` : puisque nous avons créé des variables enrichies
        """)

        df = remove_useless_col(df)
        st.code(textwrap.dedent(inspect.getsource(remove_useless_col)), language = 'python')
        # affichage des colonnes restantes
        if st.checkbox("Afficher les colonnes restantes", value = False):
            st.write(df.columns.tolist())

        with tab2:
            st.subheader("Préparation des données pour la modélisation")
            st.markdown("""
            Maintenant que les données sont nettoyées, on va les préparer pour la modélisation.
            Les étapes suivantes vont être réalisées :
            - On sépare le jeu de données en jeu d'entraintement et de test
            - On stocke les variables catégorielles et numériques dans des variables distinctes
            - On impute les transformations nécessaires aux variables catégorielles avec `SimpleImputer()` sur le most frequent 
            - On impute les transformations nécessaires aux variables numériques avec `SimpleImputer()` en utilisant la médiane car nous avons trop d'outliers
            - On fait une mise à l'échelle des variables numériques avec `StandardScaler()` 
            - On encode les variables catégorielles avec `OneHotEncoder()`
            """)

                # Chargement des données prêtes
            X_train_final, X_test_final, y_train, y_test = ready_to_process_data()

            #  Affichage des dimensions du jeu de données
            st.subheader("Dimensions du jeu de données :")
            st.markdown(f"""
            - X_train shape : `{X_train_final.shape}`
            - X_test shape : `{X_test_final.shape}`
            """)

            # Affichage des données prêtes pour la modélisation
            st.subheader("Aperçu des données d'entraînement prêtes pour la modélisation :")
            if st.checkbox("Afficher un aperçu des données modélisées", value = False):
                st.dataframe(X_train_final.head(), use_container_width = True)
                st.write("Variable cible (y_train) :")
                st.dataframe(y_train.head(), use_container_width = True)