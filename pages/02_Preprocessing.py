import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer              # Importe os imputers desejados.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder                       # Idem para os encoders.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, Normalizer, RobustScaler # Idem para os scalers.
from sklearn.preprocessing import PolynomialFeatures                                # Idem para os aumentadores de dados.
from sklearn.inspection import permutation_importance                               # Para medir a importância de cada feature.
from sklearn.tree import DecisionTreeRegressor                                      # Um estimador de Machine Learning.
from sklearn.decomposition import PCA                                               # Outro estimador de ML.
from statsmodels.stats.outliers_influence import variance_inflation_factor          # Reduz o número de features.
import matplotlib.pyplot as plt                                                     # Para fazer gráficos.
import seaborn as sns                                                               # Para fazer gráficos.
import sweetviz as sv                                                               # Para fazer a EDA
import streamlit.components.v1 as components                                        # Para fazer a EDA
from imblearn.over_sampling import SMOTE                                            # Para sintetizar dados

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
except:                                                                             # Caso contrário ignore.
    pass

colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))          # Separe as colunas numéricas das categoricas para facilitar depois.
colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

preprocessing1, preprocessing2, preprocessing3, preprocessing4, preprocessing5, preprocessing6, preprocessing7, preprocessing8, preprocessing9, preprocessing10 = st.tabs(['Preenchimento de Dados', 'Codificação de Dados', 'Escalonamento de Dados', 'Aumento de Dados', 'Redução de Dados', 'Importância de Dados', 'Correlações', 'Análise Bivariada', 'EDA - Análise Exploratória de Dados - Depois', 'Síntese de Dados com SMOTE'])

with preprocessing1:
    st.subheader("Preenchimento de Dados")

    preprocessing1_1, preprocessing1_2, preprocessing1_3 = st.columns(3)

    with preprocessing1_1:
        lista_imputers = ['Nenhum', SimpleImputer(strategy='median'), KNNImputer(), IterativeImputer(initial_strategy='median')]
        imputer_selecionado = st.selectbox('Selecione o Imputer:', lista_imputers)  # Escolha um imputer da lista.

        if imputer_selecionado == 'Nenhum':                                         # Se o imputer for 'Nenhum', não faça nada.
            pass
        else:                                                                       # Caso contrário transforme os dados.
            df[colunas_numericas] = imputer_selecionado.fit_transform(df[colunas_numericas])    # Ele transforma no formato Numpy.
            df = pd.DataFrame(df, columns=df.columns)                               # Converta de volta para o formato Pandas.

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing2:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Codificação de Dados")

    preprocessing2_1, preprocessing2_2, preprocessing2_3 = st.columns(3)

    with preprocessing2_1:                                                                  
        lista_encoders = ['Nenhum', LabelEncoder(), OneHotEncoder(drop='first')]    # ATENÇÃO: drop='first' <---- Super importante!
        encoder_selecionado = st.selectbox('Selecione o Encoder:', lista_encoders)

        if encoder_selecionado == 'Nenhum':
            pass
        elif isinstance(encoder_selecionado, LabelEncoder):                         # Se o encoder selecionado for o LabelEndoder, faça de um jeito.
            for col in colunas_categoricas:                                         # Para cada coluna de categorias...
                df[col] = encoder_selecionado.fit_transform(df[col])                # ... aplique o encoder selecionado.
        else:                                                                       # Caso contrário, faça de outro jeito.
            dados_encodados = encoder_selecionado.fit_transform(df[colunas_categoricas])    # transforme as colunas categóricas todas de uma vez.
            colunas = encoder_selecionado.fit(df[colunas_categoricas]).get_feature_names_out(colunas_categoricas) # Peça para ele recuperar os nomes das colunas.
            df_encodada = pd.DataFrame(dados_encodados.toarray(), columns=colunas)  # Converta para o formato Pandas
            df = pd.concat([df[colunas_numericas].reset_index(drop=True), df_encodada.reset_index(drop=True)], axis=1) # Junte com as colunas numéricas para recompor a df completa.
        
        df_colunas_categoricas = pd.DataFrame(colunas_categoricas, columns=['Cat'])   # Salve as colunas que foram encodadas para poder restaurá-las depois
        conn = sqlite3.connect('unip_data_science.sqlite')
        df_colunas_categoricas.to_sql('colunas_categoricas', conn, if_exists='replace', index=False)   # O Pandas só salva dataframes, por isso a conversão.
        conn.close()
        

    colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))          # Ao final, atualize o que é número e o que é classe.
    colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing3:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Escalonamento de Dados")

    preprocessing3_1, preprocessing3_2, preprocessing3_3 = st.columns(3)

    with preprocessing3_1:
        lista_scalers = ['Nenhum', StandardScaler(), MinMaxScaler(), QuantileTransformer(), Normalizer(), RobustScaler()]
        scaler_selecionado = st.selectbox('Selecione o Scaler:', lista_scalers)

    if scaler_selecionado == 'Nenhum':
        pass
    else:
        dados_escalonados = scaler_selecionado.fit_transform(df[colunas_numericas])
        colunas = scaler_selecionado.fit(df[colunas_numericas]).get_feature_names_out(colunas_numericas) # Recupere os nomes das colunas.
        df_escalonada = pd.DataFrame(dados_escalonados, columns=colunas)        # Converta para o formato Pandas
        df = pd.concat([df[colunas_categoricas].reset_index(drop=True), df_escalonada.reset_index(drop=True)], axis=1)

    colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))          # Ao final, atualize o que é número e o que é classe.
    colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing4:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Aumento de Dados")

    preprocessing4_1, preprocessing4_2, preprocessing4_3 = st.columns(3)

    with preprocessing4_1:
        lista_augmentators = ['Nenhum', PolynomialFeatures()]
        augmentator_selecionado = st.selectbox('Selecione o Aumentator:', lista_augmentators)

        if augmentator_selecionado == 'Nenhum':
            pass
        else:
            dados_aumentados = augmentator_selecionado.fit_transform(df[colunas_numericas])
            colunas = augmentator_selecionado.fit(df[colunas_numericas]).get_feature_names_out(colunas_numericas) # Recupere os nomes das colunas.
            df_aumentada = pd.DataFrame(dados_aumentados, columns=colunas)          # Converta para o formato Pandas
            df = pd.concat([df[colunas_categoricas].reset_index(drop=True), df_aumentada.reset_index(drop=True)], axis=1)    # Junte com a df anterior.

    colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))  # Ao final, atualize o que é número e o que é classe.
    colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing5:                                                                # Agora a estrutura é apenas parecida às anteriores.
    st.subheader("Redução de Dados")

    preprocessing5_1, preprocessing5_2, preprocessing5_3 = st.columns(3)

    with preprocessing5_1:
        coluna_y = [st.selectbox('Selecione o alvo (Variável Dependente):', df.columns)]

    y = df[coluna_y]
    X = df.drop(coluna_y, axis=1)
    colunas_numericas_que_sobraram = list(set(colunas_numericas) - set(list(coluna_y)))

    with preprocessing5_2:
        lista_redutores = ['Nenhum', 'Multicolinearidade', 'PCA']
        redutor_selecionado = st.selectbox('Selecione o Redutor:', lista_redutores)

        if redutor_selecionado == 'Nenhum':
            lista_features_selecionados = colunas_numericas_que_sobraram
        
        elif redutor_selecionado == 'Multicolinearidade':
            vif = pd.DataFrame()
            vif["Feature"] = colunas_numericas_que_sobraram
            vif['Variance_Inflation_Factor'] = [variance_inflation_factor(X[colunas_numericas_que_sobraram].values, i) for i in range(len(colunas_numericas_que_sobraram))]
            vif = vif.sort_values(by='Variance_Inflation_Factor', ascending=False)
            vif_selecionado = vif[vif['Variance_Inflation_Factor'] <= 10]
            lista_features_selecionados = vif_selecionado["Feature"].tolist()
            df = df[colunas_categoricas + lista_features_selecionados + coluna_y]

        elif redutor_selecionado == 'PCA':
            pca = PCA(n_components=5).fit(X)                                                # Reduza X para 5 colunas ou menos.
            feature_names = list(X.columns)                                                 # Salve os nomes das colunas originais.
            component_names = []                                                            # Crie uma lista em branco para os nomes das colunas do PCA.
            for i, component in enumerate(pca.components_):                                 # Para cada coluna e componente do PCA...
                top_features_idx = np.argsort(np.abs(component))[::-1][:3]                  # ... pegue os 3 principais componentes.
                top_features = [feature_names[idx] for idx in top_features_idx]             # Extraia os nomes.
                component_name = f"PCA_{i+1}_({' & '.join(top_features)})"                  # Junte os nomes como sufixo do PCA.
                component_names.append(component_name)                                      # Junte os sufixos aos nomes que a função gera.
            
            clf = pca.transform(X)                                                          # Aplique o classifier clf à dataframe X.
            df_pca = pd.DataFrame(clf, columns=component_names)                             # Converta para Pandas.
            df = pd.concat([df[coluna_y].reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1)    # Junte com a df anterior.
            explained_variance_ratio = pca.explained_variance_ratio_                        # Diga qual componente mais afeta a redução.
            explained_variance_df = pd.DataFrame({                                          # Salve os nomes e 
                'Principal Component': component_names,
                'Explained Variance Ratio': explained_variance_ratio
            })

    colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))          # Ao final, atualize o que é número e o que é classe.
    colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

    if redutor_selecionado == 'Multicolinearidade':
        st.write("### Variance Inflation Factor")
        try:
            st.write(vif)
        except:
            pass

    if redutor_selecionado == 'PCA':
        st.write("### Variância dos Componentes do PCA")
        try:
            st.dataframe(explained_variance_df)
        except:
            pass
    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing6:                                                                # Mesma coisa ... de novo.
    st.subheader("Importância de Dados")

    X = df.drop(coluna_y, axis=1)                                                   # Mudou a df, renove o X.
    y = df[coluna_y]
    
    colunas_numericas_X = list(set(colunas_numericas) - set(coluna_y))
    colunas = list(set(df.columns) - set(coluna_y))

    clf = DecisionTreeRegressor().fit(X[colunas_numericas_X], y)
    importancia = permutation_importance(clf, X[colunas_numericas_X], y)

    importance_df = pd.DataFrame({
        'Feature': colunas_numericas_X,
        'Importância Média': importancia.importances_mean,
    })
    importance_df = importance_df.sort_values(by='Importância Média', ascending=False)

    fig1 = plt.figure(figsize=(4, 3))
    sns.barplot(data=importance_df, x='Importância Média', y='Feature', palette='viridis')
    plt.title('Importância Média dos Features/Poder de Previsão')
    plt.xlabel('Importância Média')
    plt.ylabel('Feature')
    # plt.tight_layout()
    st.pyplot(fig1, use_container_width=True, clear_figure=False)             # clear_figure=False não deixa o Streamlit apagar a figura.

    st.write("### Matriz de Importâncias")
    st.dataframe(importance_df)


with preprocessing7:
    st.subheader("Correlações")

    correlation_matrix = df[colunas_numericas].corr(method='spearman')
    fig2 = plt.figure(figsize=(4, 3))
    sns.heatmap(correlation_matrix, annot=False, annot_kws={"fontsize": 2}, cmap='viridis')
    plt.title('Matriz de Correlações')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True, clear_figure=False)             # clear_figure=False não deixa o Streamlit apagar a figura.


with preprocessing8:
    st.subheader("Análise Bivariada")

    preprocessing8_1, preprocessing8_2, preprocessing8_3, preprocessing8_4 = st.columns(4)

    with preprocessing8_1:
        eixo_x = st.selectbox('Eixo x:', colunas_numericas)

    with preprocessing8_2:
        eixo_y = st.selectbox('Eixo y:', colunas_numericas)        

    with preprocessing8_3:
        cor = st.selectbox('Cor:', df.columns)                                    # Fica como opção escolher a cor aqui ou na sidebar.

    with preprocessing8_4:
        grid = st.button('Gerar grid')                

    fig3 = plt.figure(figsize=(5, 3))
    sns.scatterplot(data=df, x=eixo_x, y=eixo_y, hue=cor, palette='viridis')
    titulo = f'Relação {eixo_x} x {eixo_y}'
    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.legend(title=cor)
    plt.legend(title=cor, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, bbox_transform=plt.gcf().transFigure)
    st.pyplot(fig3, use_container_width=True, clear_figure=False)      

    if grid:
        g = sns.FacetGrid(df,                                               # A figura será um grid de mini-gráficos, com os dados da df
                          col=cor,                                          # Cada coluna do FacetGrid será uma cor. 
                          col_wrap=3,                                       # 3 Colunas no grid.
                          sharex=False,                                     # Não compartilhe o eixo x entre as figuras.
                          sharey=False)                                     # Cada figura tem o seu próprio eixo y.
        g.map_dataframe(sns.scatterplot, x=eixo_x, y=eixo_y)                # Os mini-gráficos serão gráficos de pontos 2D.
        g.set_titles(f'{cor} = {{col_name}}')                               # O título dos mini-gráficos será a cor.
        g.set_axis_labels(eixo_x, eixo_y)                                   # Os nomes dos eixos serão as variáveis que os compõem.
        st.pyplot(plt.gcf(),                                                # Streamlit, mostre a figura.
                  use_container_width=True,                                # Não manipule o tamanho da figura para ocupar a largura da tela.
                  clear_figure=False)                                       # Não apague a figura depois de criá-la.


with preprocessing9:
    
    preprocessing9_1, preprocessing9_2, preprocessing9_3 = st.columns(3)

    st.subheader("EDA - Análise Exploratória de Dados - Depois")

    if st.button('Gerar Relatório de EDA'):

        report = sv.analyze(df)                                                     # Use o Sweetviz para gerar o relatório.
        report.show_html("eda_depois.html", open_browser=False)                     # Salve o .html no diretório local.

        with open("eda_depois.html", "r") as f:                                     # Leia "r" o .html do diretório local.
            html_content = f.read()

        components.html(html_content, height=800, scrolling=True)                   # Mostre o .html no Streamlit.


if st.sidebar.button('Salvar Dados Pré-Processados'):

    conn = sqlite3.connect('unip_data_science.sqlite')                              # Salve em sqlite.
    df.to_sql('df_transformada', conn, if_exists='replace', index=False)
    conn.close()


with preprocessing10:

    st.subheader("Síntese de Dados com SMOTE")
    st.markdown('ATENÇÃO: A coluna escolhida precisa ter sido originalmente categórica e posteriormente encodada, e as colunas numéricas não podem estar escalonadas.')

    preprocessing10_1, preprocessing10_2, preprocessing10_3 = st.columns(3)

    with preprocessing10_1:
        smote = st.selectbox('Selecione o Alvo para o SMOTE:', df.columns)

    # O SMOTE procura 5 integrantes (k neighbours) daquele grupo. Portanto, se a coluna não tiver no mínimo 5 linhas daquele grupo, o SMOTE não vai funcionar. 
    df_smote = df.groupby(smote).filter(lambda x: len(x) > 5)                    # Para excluir linhas contendo grupos com menos de 5 integrantes.

    y = pd.DataFrame(df_smote[smote], columns=[smote])
    X = df_smote.drop(columns=[smote], axis=1)

    try:
        X_res, y_res = SMOTE(sampling_strategy='minority').fit_resample(X, y)

        st.markdown(f'Base de Dados Antes {df.shape}')
        st.dataframe(df)

        df_smote = pd.DataFrame(X_res)
        df_smote[smote] = y_res

        st.markdown(f'Base de Dados Depois (SMOTE) {df_smote.shape}')
        st.dataframe(df_smote)

        with preprocessing10_2:
            if st.button('Exportar Base de Dados Ampliada com SMOTE para .csv'):
                df_smote.to_csv('df_smote.csv', index=False)
    except:
        pass
