import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import sqlite3
from itertools import product
import pickle
from datetime import datetime
import time
import io

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer              # Importe os imputers desejados.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder      # Encoders.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, Normalizer, RobustScaler   # Scalers.
from sklearn.preprocessing import PolynomialFeatures                                # Ampliadores.
from sklearn.preprocessing import FunctionTransformer                               # Para converter fórmulas para o formato Sklearn.

from sklearn.dummy import DummyRegressor                                            # Modelo inerte para referência.
from sklearn.linear_model import LinearRegression                                   # Modelos do Scikit-Learn para regressão.
from xgboost import XGBRegressor                                                    # Modelos do XGBoost para regressão.
from sklearn.ensemble import RandomForestRegressor                                  # Várias árvores de decisão na regressão.
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

from sklearn import set_config                                                      # Para configurar o pipeline plotado.
from sklearn.utils import estimator_html_repr                                       # Para plotar o pipeline.

import seaborn as sns                                                               # Para fazer gráficos.
import matplotlib.pyplot as plt                                                     # Para configurar gráficos.



st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados na Prática")
st.subheader("Prof. Thiago Gatti")

try:                                                                                # Se existir uma base de dados em sql carrege.
    conn = sqlite3.connect('unip_data_science.sqlite')
    df = pd.read_sql('select * from df', conn)
    log = pd.read_sql('select * from log', conn)
    conn.close()
    del conn
except:                                                                             # Caso contrário pare.
    pass

try:
    conn = sqlite3.connect('unip_data_science.sqlite')
    log = pd.read_sql('select * from log', conn)
    conn.close()
    del conn
except:
    pass

colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))      # Separe as colunas numéricas das categoricas para facilitar depois.
colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))


ml1, ml2, ml3, ml4 = st.tabs(['Treinamento e Avaliação', 'Explicação', 'Forecast', 'Log'])

with ml1:

    st.subheader("Treinamento e Avaliação")

    set_config(transform_output = 'pandas')

    ml1_1, ml1_2, ml1_3, ml1_4, ml1_5 = st.columns(5)

    with ml1_1:

        # categorias = df.select_dtypes(include=['object', 'category']).columns       # Remova as colunas com menos de 3 clases, para o SMOTE.
        # unique_counts = df[categorias].apply(lambda x: x.nunique())                 # Nessa versão o SMOTE foi desativado porque ele pode ser feito por fora do pipe, criando um .csv e por causa dos erros.
        # columns_to_drop = unique_counts[unique_counts < 3].index
        # df.drop(columns=columns_to_drop, inplace=True)
        target = st.selectbox('Target:', df.columns, index=14)                                # Das colunas que sobraram escolha o alvo y.

    with ml1_2:
        encoders_dict = {
            'Nenhum': 'passthrough',
            'Ordinal': OrdinalEncoder(handle_unknown='use_encoded_value',           # Alternativa ao LabelEncoder, que sairá de linha. 
                                      unknown_value=-1,                             # Atribua -1 aos valores não identificados.
                                      encoded_missing_value=-2,                     # Atribua -2 aos campos nulos.
                                      ),                     
            'OneHot': OneHotEncoder(drop='first',                                   # Evita o Dummy Variable Trap, que evita overfitting.
                                    sparse_output=False,                            # Evita o erro: ValueError: For a sparse output, all columns should be a numeric or convertible to a numeric.
                                    handle_unknown='ignore',                        # Evita o erro: ValueError: Found unknown categories [80, ...] in column 2 during transform
                                    ),
        }
        encoder_key = st.selectbox('Encoder', options=list(encoders_dict.keys()), index=1)
        encoder = encoders_dict[encoder_key]

        imputers_dict = {                                                           # O dicionário é para apresentar nomes mais amistosos.
            'Nenhum': 'passthrough',
            'Simple Imputer': SimpleImputer(strategy="most_frequent"),              # Evita o erro: ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: 'Sales'
        }
        imputer_key = st.selectbox('Imputer', options=list(imputers_dict.keys()), index=1)
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
        scaler_key = st.selectbox('Scaler', options=list(scalers_dict.keys()), index=1)
        scaler = scalers_dict[scaler_key]

    with ml1_4:
        # smote = st.checkbox('SMOTE')
        poly = st.checkbox('Polynomial Features')
        reduction = st.checkbox('Feature Redution')

    with ml1_5:
        estimators_dict = {                                                         
            'Regressão Dummy': DummyRegressor(strategy='median'),                   # A referência é prever a mediana.
            'Regressão Linear': LinearRegression(),                                 # O modelo mais simples de regressão.
            'Regressão XGBoost': XGBRegressor(random_state=10),                     # Um modelo de regressão que "pensa" como classificação.
            'Floresta Randômica para Regressão': RandomForestRegressor(),           # Um modelo de classificação usado em regressão com muito sucesso.
            'Árvore de Decisão': DecisionTreeClassifier(),                          # O modelo mais tradicional de classificação.
            'Rede Neural para Regressão MLP': MLPRegressor(),                       # Um modelo de Rede-Neural para regressão.
            'Rede Neural para Classificação MLP':  MLPClassifier(),                 # Um modelo de Rede-Neural para classificação.
            'Clusterização': MeanShift(),                                           # Um modelo para clusterização.
            'Detecção de Fraudes': IsolationForest(),                               # Um modelo para detecção de fraudes.
        }
        estimator_key = st.selectbox('Estimator', options=list(estimators_dict.keys()), index=1)
        estimator = estimators_dict[estimator_key]

    # if st.button('Teinar Modelo'):
    t = time.time()

    y_aux = pd.DataFrame(df[target], columns=[target])
    X = df.drop(columns=[target], axis=1)

    try:                                                                        # Se o y for número:
        y = SimpleImputer().fit_transform(y_aux).ravel()                        # Não pode vir campos faltando, .ravel() garante que y será uma 1D array.
        y = pd.DataFrame(y, columns=y_aux.columns)                            # O Sklearn transforma em formato numpy. Reverter para Pandas.
    except:                                                                     # Se for classe:
        y = df[target].replace(np.nan, 'vazio').values.ravel()                  # Troque campos faltando pela classe/texto 'vazio'.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True) # Faça o split.

    # categorias = X_train.select_dtypes(include=['object', 'category']).columns  # Para o SMOTE, do que foi separado para treino...
    # for col in categorias:                                                      # para cada coluna categórica...
    #     counts = X_train[col].value_counts()
    #     valid_labels = counts[counts >= 3].index
    #     X_train = X_train[X_train[col].isin(valid_labels)]                      # mantenha só as linhas com 3 ou mais ocorrências de cada grupo.

    # y_train = y_train.loc[X_train.index]                                        # E pegue o y com as linhas/índices que você manteve.

    colunas_categoricas = X_train.select_dtypes(include=['object', 'category']).columns
    colunas_numericas = X_train.select_dtypes(include=['number']).columns      # Resignifique o que são categorias e números porque a coluna target sai da bd.

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
            ('cat', categorical_pipeline, colunas_categoricas),
            ('num', numerical_pipeline, colunas_numericas),
        ], remainder='passthrough')

    # if smote:
    #     smote_step = ('Data Synthesis', SMOTE(sampling_strategy='minority',k_neighbors=3))
    # else:
    #     smote_step = ('Data Synthesis', 'passthrough')

    if poly:
        poly_step = ('Feature Augmentation', PolynomialFeatures())
    else:
        poly_step = ('Feature Augmentation', 'passthrough')

    if reduction:
        reduction_step = ('Feature Reduction', SelectKBest(k=5, score_func=f_regression))
    else:
        reduction_step = ('Feature Reduction', 'passthrough')


    pipeline = Pipeline(steps=[
        ('Preprocessing', preprocessor),
        # smote_step,
        poly_step,
        reduction_step,
        ('Estimation', estimator)
    ])

    # try:
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)


    # Regressão -------------------------------------------------------------------------------------------------------

    # O Sciencekit-Learn roda em numpy, então é melhor converter os dados para np para poder usar os gráficos prontos do Sklearn:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()

    # O X vai sofrendo transformações ao longo do pipe. Para acessar o X resultante é preciso refazer o trajeto do pipe.
    X_preprocessed = preprocessor.fit_transform(X_train)
    
    # if smote:
    #     X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y_train)
    # else:
    #     X_resampled, y_resampled = X_preprocessed, y_train

    if poly:
        X_poly = PolynomialFeatures().fit_transform(X_preprocessed)
    else:
        X_poly = X_preprocessed

    if reduction:
        X_reduced = SelectKBest(k=10, score_func=f_regression).fit_transform(X_poly, y_train)
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
            titulo = f'{target} Forecast, R2={r2_score(y_test, y_pred):.2f}'
            plt.title(titulo)
            plt.xlabel('Real')
            plt.ylabel('Previsto')
            st.pyplot(fig3, use_container_width=True, clear_figure=False)      

        with ml1_c4:
            ped = PredictionErrorDisplay.from_predictions(y_true=y_test, y_pred=y_pred)
            titulo = f'{target} Erro (Resíduo) de Forecast'
            plt.title(titulo)
            st.pyplot(plt, use_container_width=True, clear_figure=False)

    with st.container():

        ml1_c5, ml1_c6 = st.columns(2)

        with ml1_c5:
            fig5 = plt.figure(figsize=(5, 3))  # Adjust figsize as needed
            pipeline_steps = [step for step in pipeline.named_steps]
            plt.barh(pipeline_steps, range(len(pipeline_steps)))
            plt.yticks(range(len(pipeline_steps)), pipeline_steps)
            plt.title('Consumo de Processamento')
            st.pyplot(fig5, use_container_width=True, clear_figure=False)

        with ml1_c6:
            set_config(display='diagram')
            html_representation = estimator_html_repr(pipeline)
            components.html(html_representation, height=400)


    # st.markdown(f'##### Dependência Parcial')
    # columns = list(X_reduced.columns)
    # pairs = list(product(columns, repeat=2))
    # pairs_df = pd.DataFrame(pairs, columns=['col1', 'col2'])
    
    # fig6 = sns.FacetGrid(pairs_df, col='col1', col_wrap=3, height=3, aspect=1.5)
    # for ax, col1 in zip(fig6.axes.flat, X_reduced.columns):
    #     pdp_display = PartialDependenceDisplay.from_estimator(estimator, X_reduced, [col1])
    #     pdp_display.plot(ax=ax)
    # fig6.fig.subplots_adjust(hspace=0.4, wspace=0.3)
    # st.pyplot(fig6, use_container_width=True, clear_figure=False)



#-------------------------------------------------------------------- ajustar os nomes das variaveis aqui
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

    def weighted_accuracy(y_true, y_pred, weights):
        assert len(y_true) == len(y_pred)
        assert len(weights) == len(np.unique(y_true))
        correct = (y_true == y_pred)
        weighted_correct = np.sum(weights[y_true] * correct)
        total_weighted = np.sum(weights[y_true])
        return weighted_correct / total_weighted

    def wmpa(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        percentage_accuracy = 1 - mape        
        pa_values = percentage_accuracy(y_true, y_pred)
        return np.sum(pa_values * y_true) / np.sum(y_true)
#-------------------------------------------------------------------- ajustar os nomes das variaveis aqui





    date = datetime.now()
    elapsed = (time.time() - t)                                                         # /60 para ter em minutos.

    results_dict = {
        'Type': 'Regression',
        'Date': date.strftime('%Y/%m/%d'),
        'Elapsed Time (Seconds)': f"{elapsed:.4f}",
        'Dataframe': 'placeholder',
        'Target': target,
        'Samples': df.shape[0],
        'Features Before': df.shape[1],
        'Features After': X_reduced.shape[1],
        'Encoder': str(encoder),
        'Imputer': str(imputer),
        'Scaler': str(scaler),
        'Estimator': str(estimator),
        # 'SMOTE': str(smote),
        'Poly': str(poly),
        'Reduction': str(reduction),
        'R2': f'{r2_score(y_test, y_pred):.2f}',
        'MAE': f'{mean_absolute_error(y_test, y_pred):.2f}',
        'MSE': f'{mean_squared_error(y_test, y_pred):.2f}',
        # 'MSLE': f'{mean_squared_log_error(y_test, y_pred):.2f}',                               # ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
        'MAPE': f'{mean_absolute_percentage_error(y_test, y_pred):.2f}',
        'WMAPE': 'placeholder',
        'Weighted Accuracy': 'placeholder',
        'WMPA': 'placeholder',
        'MedAE': f'{median_absolute_error(y_test, y_pred):.2f}',
        'Max Error': f'{max_error(y_test, y_pred):.2f}',
        'Explained Variance': f'{explained_variance_score(y_test, y_pred):.2f}',
        'Cross Validation Mean': 'placeholder',
        'Cross Validation Std': 'placeholder',
        'Selected Features': str((list(X_reduced.columns))),
    }
    results = pd.DataFrame([results_dict])
    st.markdown(f'##### KPI´s')
    st.dataframe(results.T, use_container_width=True)

    st.markdown(f'##### X Treino após o Pipeline {X_reduced.shape}')
    st.dataframe(X_reduced)

    # except:
    #     pass


with st.sidebar:

    if st.button('Salvar Modelo'):
        with open('model.pkl', 'wb') as file:
            pickle.dump(pipeline, file)

        df_target = pd.DataFrame([target], columns=['Target'])
        conn = sqlite3.connect('unip_data_science.sqlite')
        df_target.to_sql('df_target', conn, if_exists='replace', index=False)   
        conn.close()
        

    if st.button('Adicionar KPI´s ao Log'):
        conn = sqlite3.connect('unip_data_science.sqlite')
        results.to_sql('log', conn, if_exists='append', index=False)
        conn.close()


    if st.button('Gerar Relatório'):
        st.markdown('placeholder')

with ml2:
    st.subheader('Explicação')

    # o shap vem aqui porque ele muda de entender se o modelo está prevendo corretamente para porque está fazendo essas previsões
    # parte 1 verificar se a previsão está correta
    # parte 2 pq fez essa previsao
    # shap da arvore https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Example%20of%20loading%20a%20custom%20tree%20model%20into%20SHAP.html
    # probabilidade de ganhar https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/League%20of%20Legends%20Win%20Prediction%20with%20XGBoost.html
    # SHAP interaction values show how interactions between two features affect the model's predictions https://chatgpt.com/c/19eabec8-d4eb-4d76-9da1-aaed6904d888
    # 



with ml3:
    st.subheader("Forecast")

    try:
        with open('model.pkl', 'rb') as file:
            pipeline = pickle.load(file)    
    except:
        st.markdown('Ainda não existe nenhum modelo salvo.') 


    ml3_1, ml3_2, ml3_3 = st.columns(3)

    X_predict = pd.DataFrame()
    
    with ml3_1:
        st.markdown('##### Categorias')
        for col in colunas_categoricas:
            labels = df[col].unique()
            value = st.selectbox(col, labels)
            X_predict[col] = [value]

    with ml3_2:
        st.markdown('##### Valores')
        for col in colunas_numericas:
            value = st.number_input(col, 0.0)
            X_predict[col] = [value]

    with ml3_3:
        conn = sqlite3.connect('unip_data_science.sqlite')
        target = pd.read_sql('select * from df_target', conn)
        conn.close()

        st.markdown(f'##### Estimativa de {target.iloc[0].tolist()[0]}')

        if st.button('Estimar'):
            estimativa = pipeline.predict(X_predict)
            st.write(estimativa)

    
    # ele precisa poder identificar qual foi o target e tirar do filtro e adicionar a previsao e ser diferente da tela de treinamento do modelo
    # pq aqui e agora, depois do modelo salvo, nao importa mais o q ficou selecionado na tela de modelo salvo
    
    # resolver a dependencia parcial quando tem reducao de features

    # reservar restaurante p 10

    # ver se era o target encoder q alterava a relação crescente p preco e qtd

    # criar mais uma guia no ml p comparar a estimatva c o ground truth e dar um percentual de acerto chamado de success

    # mostrar q se as variaveis forem independentes entao a redução vai reduzir pouco

    # fazer os kpi´s de classificação

    # p montar um cenario o cara pode ir olhando no pbi p saber quais são as faixas q mantem 50% dentro e etc
    # ou aperta botão p o ai sugerir, e ele sugere o quartil? Não, ma ideia, pq no caso de RS os valores sempre foram ruins.
    # fazer uma bd consolidada por região sem cli se a acuracidade real for ruin
    # ensinar isso, se acuracidade for ruim entao consolidar mais. mostrar o comparativo das 2.

    # https://www.youtube.com/watch?v=4Nqd5qD46BY

    # primeiro compoarar o forecast c os dados reais p ver se deu overfit -> etapa de afericao
    # depois, se vc confia no modelo, e fazer 1 slide p isso, aqui é a deixa p falar: lembra q naquela região o roi tava baixo, entao qual deve ser o preco p ele aumentar?
    # dizer q aqui é onde pode misturar c o pbi



with ml4:
    st.subheader("Log de Registros")
    try:
        st.dataframe(log)
    except:
        st.markdown('O log de registros ainda está vazio.')
