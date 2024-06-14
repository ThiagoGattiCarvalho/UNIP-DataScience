import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from itertools import product

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer              # Importe os imputers desejados.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder                     # Encoders.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, Normalizer, RobustScaler   # Scalers.
from sklearn.preprocessing import PolynomialFeatures                                # Ampliadores.
from sklearn.preprocessing import FunctionTransformer                               # Para converter fórmulas para o formato Sklearn.

from sklearn.dummy import DummyRegressor                                            # Modelo inerte para referência.
from sklearn.linear_model import LinearRegression                                   # Modelos do Scikit-Learn para regressão.
from xgboost import XGBRegressor                                                    # Modelos do XGBoost para regressão.
from sklearn.tree import DecisionTreeClassifier                                     # Modelos para classificação.
from sklearn.neural_network import MLPRegressor, MLPClassifier                      # Modelos de Rede-Neural para regressão e classificação.
from sklearn.cluster import MeanShift                                               # Modelo para clusterização.
from sklearn.ensemble import IsolationForest                                        # Modelo para detecção de fraudes.

from sklearn.compose import ColumnTransformer                                       # Para criar o preprocessador.
from imblearn.over_sampling import SMOTE                                            # Para incluir SMOTE.
from imblearn.pipeline import Pipeline                                              # Para fazer um pipeline que aceita SMOTE.
# from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression                     # Para excolher features usando o p-balue (F-Statistics).

from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap                                                                         # Para explicar as importâncias.

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, mean_absolute_percentage_error, max_error, explained_variance_score, PredictionErrorDisplay

from sklearn import set_config

import matplotlib.pyplot as plt                                                     # Para fazer gráficos.
import seaborn as sns                                                               # Para fazer gráficos.



st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados e Estudos Analíticos de Big Data")
st.subheader("Prof. Thiago Gatti")

try:                                                                                # Se existir uma base de dados em sql carrege.
    conn = sqlite3.connect('unip_data_science.sqlite')
    df = pd.read_sql('select * from df', conn)
    conn.close()
    del conn
except:                                                                             # Caso contrário pare.
    st.stop()

ml1, ml2, ml3 = st.tabs(['Treinamento e Avaliação de Modelos', 'Simulação de Cenários', 'Planejamento Estratégico'])

with ml1:

    st.subheader("Treinamento e Avaliação de Modelos")

    set_config(transform_output = "pandas")

    ml1_1, ml1_2, ml1_3, ml1_4, ml1_5 = st.columns(5)

    with ml1_1:

        categorias = df.select_dtypes(include=['object', 'category']).columns       # Remova as colunas com menos de 3 clases, para o SMOTE.
        unique_counts = df[categorias].apply(lambda x: x.nunique())
        columns_to_drop = unique_counts[unique_counts < 3].index
        df.drop(columns=columns_to_drop, inplace=True)

        target = st.selectbox('Target:', df.columns)                                # Das colunas que sobraram escolha o alvo y.
        y_aux = pd.DataFrame(df[target], columns=[target])
        X = df.drop(columns=[target], axis=1)

        try:                                                                        # Se o y for número:
            y = SimpleImputer().fit_transform(y_aux)                                # Não pode vir campos faltando.
            y = pd.DataFrame(y, columns=y_aux.columns)                              # O Sklearn transforma em formato numpy. Reverter para Pandas.
        except:                                                                     # Se for classe:
            y = df[target].replace(np.nan, 'vazio')                                 # Troque campos faltando pela classe/texto 'vazio'.

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True) # Faça o split.

        categorias = X_train.select_dtypes(include=['object', 'category']).columns  # Do que foi separado para treino...
        for col in categorias:                                                      # para cada coluna categórica...
            counts = X_train[col].value_counts()
            valid_labels = counts[counts >= 3].index
            X_train = X_train[X_train[col].isin(valid_labels)]                      # mantenha só as linhas com 3 ou mais ocorrências de cada grupo.

        y_train = y_train.loc[X_train.index]                                        # E pegue o y com as linhas/índices que você manteve.

        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
        numerical_features = X_train.select_dtypes(include=['number']).columns      # Resignifique o que são categorias e números baseado no que sobrou.
        all_features = list(categorical_features) + list(numerical_features)

    with ml1_2:
        encoders_dict = {
            'Nenhum': 'passthrough',
            'Ordinal': OrdinalEncoder(handle_unknown='use_encoded_value',           # Alternativa ao LabelEncoder, que sairá de linha. 
                                      unknown_value=-1,                             # Atribua -1 aos vazios e classes faltantes.
                                      ),                     
            'OneHot': OneHotEncoder(drop='first',                                   # Evita o Dummy Variable Trap, que evita overfitting.
                                    sparse_output=False,                            # Evita o erro: ValueError: For a sparse output, all columns should be a numeric or convertible to a numeric.
                                    handle_unknown='ignore',                        # Evita o erro: ValueError: Found unknown categories [80, ...] in column 2 during transform
                                    ),
        }
        encoder_key = st.selectbox('Encoder', options=list(encoders_dict.keys()))
        encoder = encoders_dict[encoder_key]

        imputers_dict = {                                                           # O dicionário é para apresentar nomes mais amistosos.
            'Nenhum': 'passthrough',
            'Simple Imputer': SimpleImputer(strategy="most_frequent"),              # Evita o erro: ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: 'Sales'
        }
        imputer_key = st.selectbox('Imputer', options=list(imputers_dict.keys()))
        imputer = imputers_dict[imputer_key]

    with ml1_3:
        scalers_dict = {
            'Nenhum': 'passthrough',
            'Standard': StandardScaler(with_mean=False),
            'MinMax': MinMaxScaler(),
            'Quantile': QuantileTransformer(),
            'Normalizer': Normalizer(),
            'Robust': RobustScaler(),
        }
        scaler_key = st.selectbox('Scaler', options=list(scalers_dict.keys()))
        scaler = scalers_dict[scaler_key]

    with ml1_4:
        smote = st.checkbox('SMOTE')
        poly = st.checkbox('Polynomial Features')
        reduction = st.checkbox('Feature Redution')

    with ml1_5:
        estimators_dict = {                                                         
            'Regressão Dummy': DummyRegressor(strategy='median'),                   # A referência é prever a mediana.
            'Regressão Linear': LinearRegression(),                                 # O modelo mais simples de regressão.
            'Árvore de Decisão': DecisionTreeClassifier(),                          # Um modelo muito comum de classificação.
            'Regressão XGBoost': XGBRegressor(random_state=10),                     # Um modelo de regressão que pensa como classificação.
            'Rede Neural para Regressão MLP': MLPRegressor(),                       # Um modelo de Rede-Neural para regressão.
            'Rede Neural para Classificação MLP':  MLPClassifier(),                 # Um modelo de Rede-Neural para classificação.
            'Clusterização': MeanShift(),                                           # Um modelo para clusterização.
            'Detecção de Fraudes': IsolationForest(),                               # Um modelo para detecção de fraudes.
        }
        estimator_key = st.selectbox('Estimator', options=list(estimators_dict.keys()))
        estimator = estimators_dict[estimator_key]

        categorical_pipeline = Pipeline(steps=[
            ('encoder', encoder),
            ('scaler', scaler)                                                      # Escalone depois de converter classe em inteiro para dar homogeneidade, que ajuda estimadores sensíveis a diferença de escalas, como SVMs, redes neurais e K-means.
        ])

        numerical_pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('scaler', scaler)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_pipeline, categorical_features),
                ('num', numerical_pipeline, numerical_features),
            ], remainder='passthrough')

        if smote:
            smote_step = ('SMOTE', SMOTE(sampling_strategy='minority',k_neighbors=3))
        else:
            smote_step = ('SMOTE', 'passthrough')

        if poly:
            poly_step = ('Poly', PolynomialFeatures())
        else:
            poly_step = ('Poly', 'passthrough')

        if reduction:
            reduction_step = ('Reduction', SelectKBest(k=10, score_func=f_regression))
        else:
            reduction_step = ('Reduction', 'passthrough')


        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            smote_step,
            poly_step,
            reduction_step,
            ('estimator', estimator)
        ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)


    # Regressão -------------------------------------------------------------------------------------------------------

    # O Sciencekit-Learn roda em numpy, então é melhor converter os dados para np para poder usar os gráficos prontos do Sklearn:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    # O X vai sofrendo transformações ao longo do pipe. Para acessar o X resultante é preciso refazer o trajeto do pipe.
    X_preprocessed = preprocessor.fit_transform(X_train)
    
    if smote:
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_train)
    else:
        X_resampled, y_resampled = X_preprocessed, y_train

    if poly:
        X_poly = PolynomialFeatures().fit_transform(X_resampled)
    else:
        X_poly = X_resampled

    if reduction:
        X_reduced = SelectKBest(k=10, score_func=f_regression).fit_transform(X_poly, y_resampled)
    else:
        X_reduced = X_poly
    

    with st.container():

        ml1_c1, ml1_c2 = st.columns(2)

        with ml1_c1:
            cv = cv = KFold(n_splits=5, shuffle=True)
            train_sizes, train_scores, test_scores = learning_curve(pipeline, X, y, 
                                                                    cv=cv, 
                                                                    train_sizes=np.linspace(0.1, 1.0, 5), 
                                                                    scoring='r2')
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            fig1 = plt.figure(figsize=(5, 3))
            titulo = f'Curva de Aprendizado, R2={r2_score(y_test, y_pred):.2f}'
            plt.title(titulo)
            plt.xlabel("Amostras")
            plt.ylabel('R2 Score')
            plt.grid()
            plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1, color="b")
            plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1, color="g")
            plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Training Score")
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
            plt.legend(loc="best")
            st.pyplot(fig1, use_container_width=True, clear_figure=False)

        with ml1_c2:
            importancia = permutation_importance(pipeline, X_train, y_train, random_state=42, scoring='r2')
            importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importância': importancia.importances_mean})
            importance_df.sort_values(by='Importância', ascending=False, inplace=True)

            fig2 = plt.figure(figsize=(5, 3))
            sns.barplot(data=importance_df, x='Importância', y='Feature', palette='viridis')
            plt.title('Importância Média dos Features/Poder de Previsão')
            plt.xlabel('Importância')
            plt.ylabel('Feature')
            st.pyplot(fig2, use_container_width=True, clear_figure=False)

    with st.container():

        ml1_c3, ml1_c4 = st.columns(2)

        with ml1_c3:

            fig3 = plt.figure(figsize=(5, 3))
            sns.scatterplot(x=y_test, y=y_pred, palette='viridis')
            sns.lineplot(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], color='red', linestyle='--')
            titulo = f'Forecast, R2={r2_score(y_test, y_pred):.2f}'
            plt.title(titulo)
            plt.xlabel('Real')
            plt.ylabel('Previsto')
            st.pyplot(plt.gcf(), use_container_width=True, clear_figure=False)      

        with ml1_c4:
            ped = PredictionErrorDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
            plt.title("Erro (Resíduo) de Forecast")
            st.pyplot(plt, use_container_width=True, clear_figure=False)

    results = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        # 'MSLE': mean_squared_log_error(y_test, y_pred),                                 # ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
        'MAPE': mean_absolute_percentage_error(y_test, y_pred),
        'MedAE': median_absolute_error(y_test, y_pred),
        'Max Error': max_error(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
    }
    st.markdown(f'##### KPI´s')
    st.dataframe(pd.DataFrame([results]))

    st.markdown(f'##### X Treino após o Pipeline {X_reduced.shape}')
    st.dataframe(X_reduced)

    if st.button('Gerar Relatório de Dependência Parcial'):
        
        columns = list(X_reduced.columns)
        pairs = list(product(columns, repeat=2))
        pairs_df = pd.DataFrame(pairs, columns=['col1', 'col2'])
        g = sns.FacetGrid(pairs_df, col='col1', col_wrap=3, height=3, aspect=1.5, margin_titles=True)
        for ax, (idx, row) in zip(g.axes.flat, pairs_df.iterrows()):
            col1 = row['col1']
            col2 = row['col2']
            pdp_display = PartialDependenceDisplay.from_estimator(estimator, X_reduced, [col2, (col2, col1)])
            pdp_display.plot(ax=ax)
        plt.suptitle('Dependência Parcial')
        # plt.tight_layout(h_pad=20, w_pad=20)
        g.fig.subplots_adjust(hspace=0.5, wspace=0.5)
        st.pyplot(g, use_container_width=True, clear_figure=False)





# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py