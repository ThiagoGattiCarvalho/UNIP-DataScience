import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer              # Importe os imputers desejados.
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder                     # Idem para os encoders.
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, Normalizer, RobustScaler # Idem para os scalers.
from sklearn.preprocessing import PolynomialFeatures                                # Idem para os aumentadores de dados.
from sklearn.inspection import permutation_importance                               # Para medir a importância de cada feature.
from sklearn.tree import DecisionTreeRegressor                                      # Um estimador de Machine Learning.
from sklearn.linear_model import LinearRegression                                   # Modelos do Scikit-Learn para regressão.
from sklearn.decomposition import PCA                                               # Outro estimador de ML.
from statsmodels.stats.outliers_influence import variance_inflation_factor          # Reduz o número de features.
import matplotlib.pyplot as plt                                                     # Para fazer gráficos.
from matplotlib.sankey import Sankey
import seaborn as sns                                                               # Para fazer gráficos.
from statsmodels.nonparametric.smoothers_lowess import lowess                       # Para equacionar.
from scipy import stats                                                             # Para funções estatísticas.
import sweetviz as sv                                                               # Para fazer a EDA.
import streamlit.components.v1 as components                                        # Para fazer a EDA.
from imblearn.over_sampling import SMOTE                                            # Para sintetizar dados.
import holoviews as hv
from holoviews import opts
from bokeh.io import output_file, save, show
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados na Prática")
st.subheader("Prof. Thiago Gatti")

try:                                                                                # Se existir uma base de dados em sql carrege.
    conn = sqlite3.connect('unip_data_science.sqlite')
    df = pd.read_sql('select * from df', conn)
    conn.close()
except:                                                                             # Caso contrário ignore.
    pass

colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))      # Separe as colunas numéricas das categoricas para facilitar depois.
colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

df_colunas_categoricas = pd.DataFrame(colunas_categoricas, columns=['Cat'])         # O Pandas só salva dataframes, por isso a conversão.
conn = sqlite3.connect('unip_data_science.sqlite')                                  # Salve as colunas categóricas para usá-las depois.
df_colunas_categoricas.to_sql('colunas_categoricas', conn, if_exists='replace', index=False)   
conn.close()

df2 = pd.DataFrame()                                                                # Crie uma df para os dados transformados.

with st.sidebar:
    coluna_y = [st.selectbox('Target:', df.columns, index=14)]



preprocessing1, preprocessing2, preprocessing3, preprocessing4, preprocessing5, preprocessing6, preprocessing7, preprocessing8, preprocessing9, preprocessing10, preprocessing11 = st.tabs(['Imputer', 'Encoder', 'Scaler', 'Feature Augmentation', 'Feature Reduction', 'Importância dos Features', 'Correlação', 'Análise Bivariada', 'Síntese de Dados', 'EDA - Análise Exploratória de Dados - Depois', 'Análise de Fluxo'])

with preprocessing1:
    st.subheader("Imputer")

    preprocessing1_1, preprocessing1_2, preprocessing1_3 = st.columns(3)

    with preprocessing1_1:
        lista_imputers = ['Nenhum', SimpleImputer(strategy='median'), KNNImputer(), IterativeImputer(initial_strategy='median')]
        imputer_selecionado = st.selectbox('Imputer:', lista_imputers)              # Escolha um imputer da lista.

        if imputer_selecionado == 'Nenhum':                                         # Se o imputer for 'Nenhum', não faça nada.
            df2 = df
        else:                                                                       # Caso contrário transforme os dados.
            df2[colunas_categoricas] = df[colunas_categoricas]
            df2[colunas_numericas] = imputer_selecionado.fit_transform(df[colunas_numericas])    # Ele transforma no formato Numpy.
            df2 = pd.DataFrame(df2, columns=df2.columns)                            # Converta de volta para o formato Pandas, porque o Sklearn devolve numpy.

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df2.shape})""")                               # .shape mostra quantas linhas e colunas.
        st.dataframe(df2)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing2:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Encoder")

    preprocessing2_1, preprocessing2_2, preprocessing2_3 = st.columns(3)

    with preprocessing2_1:                                                                  
        lista_encoders = ['Nenhum', OrdinalEncoder(), OneHotEncoder(drop='first')]  # ATENÇÃO: drop='first' <---- Super importante!
        encoder_selecionado = st.selectbox('Encoder:', lista_encoders)

        if encoder_selecionado == 'Nenhum':
            pass
        
        elif isinstance(encoder_selecionado, OrdinalEncoder):
            df3 = encoder_selecionado.fit_transform(df2[colunas_categoricas])
            df3 = pd.DataFrame(df3, columns=encoder_selecionado.get_feature_names_out())
            df2[colunas_categoricas] = df3

        elif isinstance(encoder_selecionado, OneHotEncoder):
            df3 = encoder_selecionado.fit_transform(df2[colunas_categoricas])
            df3 = pd.DataFrame(df3.toarray(), columns=encoder_selecionado.get_feature_names_out())
            df2 = pd.concat([df3, df2[colunas_numericas]], axis=1)
        
        
    # Atualize o que é número e o que é classe (sem alterar as listas iniciais) para os próximos transformers saberem onde se aplicar.
    colunas_numericas_modificadas = sorted(list(df2.select_dtypes(include=['number']).columns))          
    colunas_categoricas_modificadas = sorted(list(df2.select_dtypes(include=['object']).columns))

    try:                                                                            # Mostre como ficou a df, se existir a df.
        st.write(f"""##### Dataframe {df2.shape})""")                               # .shape mostra quantas linhas e colunas.
        st.dataframe(df2)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing3:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Scaler")

    preprocessing3_1, preprocessing3_2, preprocessing3_3 = st.columns(3)

    with preprocessing3_1:
        lista_scalers = ['Nenhum', StandardScaler(), MinMaxScaler(), QuantileTransformer(), Normalizer(), RobustScaler()]
        scaler_selecionado = st.selectbox('Scaler:', lista_scalers)

    if scaler_selecionado == 'Nenhum':
        pass
    
    else:
        df3 = scaler_selecionado.fit_transform(df2[colunas_numericas_modificadas])
        df3 = pd.DataFrame(df3, columns=scaler_selecionado.get_feature_names_out())
        df2 = pd.concat([df2[colunas_categoricas_modificadas], df3], axis=1)

    colunas_numericas_modificadas = sorted(list(df2.select_dtypes(include=['number']).columns))          
    # colunas_categoricas_modificadas = sorted(list(df2.select_dtypes(include=['object']).columns))

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df2.shape})""")                               # .shape mostra quantas linhas e colunas.
        st.dataframe(df2)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with preprocessing4:                                                                # Exatamente a mesma estrutura dos cases anteriores
    st.subheader("Feature Augmentation")

    preprocessing4_1, preprocessing4_2, preprocessing4_3 = st.columns(3)

    with preprocessing4_1:
        lista_augmentators = ['Nenhum', PolynomialFeatures()]
        augmentator_selecionado = st.selectbox('Augmentator:', lista_augmentators)

        if augmentator_selecionado == 'Nenhum':
            pass
        
        else:
            df3 = augmentator_selecionado.fit_transform(df2[colunas_numericas_modificadas])
            df3 = pd.DataFrame(df3, columns=augmentator_selecionado.get_feature_names_out())
            df2 = pd.concat([df2[colunas_categoricas_modificadas], df3], axis=1)

    colunas_numericas_modificadas = sorted(list(df2.select_dtypes(include=['number']).columns))
    # colunas_categoricas_modificadas = sorted(list(df2.select_dtypes(include=['object']).columns))

    try:
        st.write(f"""##### Dataframe {df2.shape})""")
        st.dataframe(df2)
    except:
        st.stop()


with preprocessing5:                                                                # Agora a estrutura é apenas parecida às anteriores.
    st.subheader("Feature Reduction")

    preprocessing5_1, preprocessing5_2, preprocessing5_3 = st.columns(3)

    y = df2[coluna_y]
    X = df2.drop(coluna_y, axis=1)
    
    colunas_numericas_que_sobraram = list(set(colunas_numericas_modificadas) - set(list(coluna_y)))

    with preprocessing5_1:
        lista_redutores = ['Nenhum', 'Multicolinearidade', 'PCA']
        redutor_selecionado = st.selectbox('Redutor:', lista_redutores)

        if redutor_selecionado == 'Nenhum':
            lista_features_selecionados = colunas_numericas_que_sobraram
        
        elif redutor_selecionado == 'Multicolinearidade':
            vif = pd.DataFrame()
            vif["Feature"] = colunas_numericas_que_sobraram
            vif['Variance_Inflation_Factor'] = [variance_inflation_factor(X[colunas_numericas_que_sobraram].values, i) for i in range(len(colunas_numericas_que_sobraram))]
            vif = vif.sort_values(by='Variance_Inflation_Factor', ascending=False)
            vif_selecionado = vif[vif['Variance_Inflation_Factor'] <= 10]
            lista_features_selecionados = vif_selecionado["Feature"].tolist()
            df2 = df2[colunas_categoricas_modificadas + lista_features_selecionados + coluna_y]

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
            df2 = pd.concat([df2[coluna_y], df_pca], axis=1)                                # Junte com a df anterior.
            explained_variance_ratio = pca.explained_variance_ratio_                        # Diga qual componente mais afeta a redução.
            explained_variance_df = pd.DataFrame({                                          # Salve os nomes e 
                'Principal Component': component_names,
                'Explained Variance Ratio': explained_variance_ratio
            })

    colunas_numericas_modificadas = sorted(list(df2.select_dtypes(include=['number']).columns))

    try:
        st.write(f"""##### Dataframe {df2.shape})""")
        st.dataframe(df2)
    except:
        st.stop()

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


with preprocessing6:                                                                        # Mesma coisa ... de novo.
    st.subheader("Importância dos Features")

    X = df2.drop(coluna_y, axis=1)                                                          # Mudou a df, renove o X.
    y = df2[coluna_y]

    colunas_numericas_X = list(set(colunas_numericas_modificadas) - set(coluna_y))
    colunas_numericas_X2 = list(set(colunas_numericas) - set(coluna_y))

    try:
        try:                                                                                    # Tente com um classifier.
            clf = DecisionTreeRegressor().fit(X[colunas_numericas_X], y)
            importancia = permutation_importance(clf, X[colunas_numericas_X], y)
        except:                                                                                 # Se não der, use um regressor.
            clf = LinearRegression().fit(X[colunas_numericas_X], y)
            importancia = permutation_importance(clf, X[colunas_numericas_X], y)
    except:
        try:
            clf = DecisionTreeRegressor().fit(X[colunas_numericas_X2], y)
            importancia = permutation_importance(clf, X[colunas_numericas_X2], y)
        except:
            clf = LinearRegression().fit(X[colunas_numericas_X2], y)
            importancia = permutation_importance(clf, X[colunas_numericas_X2], y)



    importance_df = pd.DataFrame({
        'Feature': colunas_numericas_X,
        'Importância Média': importancia.importances_mean,
    })
    importance_df = importance_df.sort_values(by='Importância Média', ascending=False)

    fig1 = plt.figure(figsize=(4, 3))
    sns.barplot(data=importance_df, x='Importância Média', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Importância Média dos Features/Poder de Previsão')
    plt.xlabel('Importância Média')
    plt.ylabel('Feature')
    st.pyplot(fig1, use_container_width=True, clear_figure=False)             # clear_figure=False não deixa o Streamlit apagar a figura.

    st.write("### Matriz de Importâncias")
    st.dataframe(importance_df)


with preprocessing7:
    st.subheader("Correlação")

    correlation_matrix = df2[colunas_numericas_modificadas].corr(method='spearman')
    fig2 = plt.figure(figsize=(4, 3))
    sns.heatmap(correlation_matrix, annot=True, annot_kws={"fontsize": 5}, cmap='viridis')
    plt.title('Matriz de Correlação')
    st.pyplot(fig2, use_container_width=True, clear_figure=False)             # clear_figure=False não deixa o Streamlit apagar a figura.


with preprocessing8:
    st.subheader("Análise Bivariada")

    preprocessing8_1, preprocessing8_2, preprocessing8_3, preprocessing8_4 = st.columns(4)

    with preprocessing8_1:
        try:
            eixo_x = st.selectbox('Eixo x:', colunas_numericas_modificadas, index=8)
        except:
            eixo_x = st.selectbox('Eixo x:', colunas_numericas_modificadas)

    with preprocessing8_2:
        try:
            eixo_y = st.selectbox('Eixo y (MAXIMIZAR):', colunas_numericas_modificadas, index=5)        
        except:
            eixo_y = st.selectbox('Eixo y (MAXIMIZAR):', colunas_numericas_modificadas)        

    with preprocessing8_3:
        try:
            cor = st.selectbox('Cor:', list(df2.columns), index=2)                                    # Fica como opção escolher a cor aqui ou na sidebar.
        except:
            cor = st.selectbox('Cor:', list(df2.columns), index=2)
            
    with preprocessing8_4:
        grid = st.button('Gerar grid')                

    fig3 = plt.figure(figsize=(5, 3))
    sns.scatterplot(data=df2, x=eixo_x, y=eixo_y, hue=cor, palette='viridis')
    sns.kdeplot(data=df2, x=eixo_x, y=eixo_y, hue=cor, palette='viridis', levels=1, bw_adjust=1)
    titulo = f'Relação {eixo_x} x {eixo_y}'
    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.legend(title=cor)
    # plt.legend(title=cor, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, bbox_transform=plt.gcf().transFigure)
    st.pyplot(fig3, use_container_width=True, clear_figure=False)      

    if grid:                                                                # Opção usando o FacetGrid do Seaborn.
    #     g = sns.FacetGrid(df2,                                              # A figura será um grid de mini-gráficos, com os dados da df
    #                       col=cor,                                          # Cada coluna do FacetGrid será uma cor. 
    #                       col_wrap=3,                                       # 3 Colunas no grid.
    #                       sharex=False,                                     # Não compartilhe o eixo x entre as figuras.
    #                       sharey=False,                                     # Cada figura tem o seu próprio eixo y.
    #                       aspect=1.5,
    #                       )                                     
    #     g.map_dataframe(sns.regplot,                                        # Os mini-gráficos serão gráficos de pontos 2D de regressão.
    #                     x=eixo_x, 
    #                     y=eixo_y,
    #                     scatter=True,                                       # Garante que os pontos serão plotados junto com a linha de tendência. 
    #                     fit_reg=True,                                       # Plota a linha de regressão de melhor fit (R2).
    #                     ci=95,                                              # Intervalo de Confiança da linha de tendência de 90%.
    #                     label='Tendência com 95% Confiança',                # Label da linha de tendência.         
    #                     )                    
    #     g.set_titles(f'{cor} = {{col_name}}')                               # O título dos mini-gráficos será a cor.
    #     g.set_axis_labels(eixo_x, eixo_y)                                   # Os nomes dos eixos serão as variáveis que os compõem.
    #     # g.add_legend()                                                      # Para mostrar a legenda contendo o label da linha de tendência.
    #     plt.suptitle('Tendência com 95% Confiança', y=1.05, fontsize=20)
    #     st.pyplot(g,                                                     # Streamlit, mostre a figura.
    #               use_container_width=True,                                 # Não manipule o tamanho da figura para ocupar a largura da tela.
    #               clear_figure=False)                                       # Não apague a figura depois de criá-la.

        df3 = np.linalg.norm(df2[colunas_numericas], axis=0)                    # Use o numpy para normalizar e desnormalizar porque o Normalizer() do Sklearn não tem inverse_transform().
        df4 = df2[colunas_numericas] / df3
        df4.columns = colunas_numericas
        df5 = pd.concat([df2[colunas_categoricas], df4], axis=1)

        df6 = pd.DataFrame()
        nrows = len(pd.unique(df2[cor]))
        fig4, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(14, nrows * 2.5))
        for i,j in zip(range(nrows), pd.unique(df2[cor])):
            
            df7 = df2[df2[cor] == j]
            
            sns.regplot(data=df7, x=eixo_x, y=eixo_y, scatter=True, fit_reg=True, ci=95, label=j, ax=axes[i, 0], line_kws={'color': 'purple'})
            axes[i, 0].set_title('Tendência, 95% IC')
            axes[i, 0].set_xlabel(eixo_x)
            axes[i, 0].set_ylabel(eixo_y)
            axes[i, 0].legend()

            df6 = df5[df5[cor] == j]
            df6.sort_values(by=eixo_x, inplace=True)                            # O LOWESS requer ordenação.
            smoothed = lowess(df6[eixo_y], df6[eixo_x], frac=0.5)
            smoothed_x = smoothed[:, 0]
            smoothed_y = smoothed[:, 1]
            max_y = np.argmax(smoothed_y)
            best_x_normalized = smoothed_x[max_y]
            idx_eixo_x = colunas_numericas.index(eixo_x)                        # Ache o indíce do melhor valor
            best_x = best_x_normalized * df3[idx_eixo_x]                        # Procure na base original pelo índice.

            nearest_idx = np.abs(smoothed_x - best_x_normalized).argmin()
            best_y_normalized = smoothed_y[nearest_idx]
            best_y = best_y_normalized * df3.mean()

            error = df6[eixo_y] - smoothed_y
            std = np.std(error)                                                 # Desvio padrão std é propriedade na Normal e não das outras distribuições. Porém, os dados foram normalizados.
            ci = 0.95
            margin_of_error = stats.norm.ppf((1 + ci) / 2) * std
            upper_bound = smoothed_y + margin_of_error
            lower_bound = smoothed_y - margin_of_error

            sns.regplot(data=df6, x=eixo_x, y=eixo_y, fit_reg=False, ax=axes[i, 1])
            sns.lineplot(data=df6, x=smoothed_x, y=smoothed_y, label='LOWESS', ax=axes[i, 1], color= 'orange')
            axes[i, 1].fill_between(smoothed_x, lower_bound, upper_bound, color='orange', alpha=0.3)
            axes[i, 1].axvline(x=best_x_normalized, color='green', linestyle='--', label=f'Best {eixo_x}')
            axes[i, 1].set_title(f'Normalização e Best-Fit, 95% IC')
            axes[i, 1].set_xlabel(f'{eixo_x} Normalizado')
            axes[i, 1].set_ylabel(f'{eixo_y} Normalizado')
            axes[i, 1].legend()

            kde = stats.gaussian_kde(df7[eixo_y])
            y_values = np.linspace(df7[eixo_y].min(), df7[eixo_y].max(), 100)
            pdf_values = kde(y_values)
            # interval_mask = (y_values >= min_valor) & (y_values <= max_valor)
            # pdf = np.sum(pdf_values[interval_mask]) / np.sum(pdf_values)

            sns.histplot(data=df7, x=eixo_y, bins=10, kde=False, alpha=0.1, element="step", ax=axes[i, 2])
            axes2 = axes[i, 2].twinx()
            sns.lineplot(x=y_values, y=pdf_values, ax=axes2)
            label = f'Best {eixo_y}={best_y:.2f}'
            axes[i, 2].axvline(x=best_y, color='green', linestyle='--', label=label)
            axes[i, 2].set_title(f'{eixo_y} KDE, 95% IC')
            axes[i, 2].set_xlabel(eixo_y)
            axes[i, 2].set_ylabel('Contagem')
            axes2.set_ylabel('')
            axes2.set_yticks([])        
            axes[i, 2].legend()

            kde = stats.gaussian_kde(df7[eixo_x])
            x_values = np.linspace(df7[eixo_x].min(), df7[eixo_x].max(), 100)
            pdf_values = kde(x_values)
            # interval_mask = (x_values >= min_valor) & (x_values <= max_valor)
            # pdf = np.sum(pdf_values[interval_mask]) / np.sum(pdf_values)

            sns.histplot(data=df7, x=eixo_x, bins=10, kde=False, alpha=0.1, element="step", ax=axes[i, 3])
            axes2 = axes[i, 3].twinx()
            sns.lineplot(x=x_values, y=pdf_values, ax=axes2)
            label = f'Best {eixo_x}={best_x:.2f}'
            axes[i, 3].axvline(x=best_x, color='green', linestyle='--', label=label)
            axes[i, 3].set_title(f'{eixo_x} KDE, 95% IC')
            axes[i, 3].set_xlabel(eixo_x)
            axes[i, 3].set_ylabel('Contagem')
            axes2.set_ylabel('')
            axes2.set_yticks([])        
            axes[i, 3].legend()

        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True, clear_figure=False)


with preprocessing9:

    st.subheader("Síntese de Dados")
    st.markdown('ATENÇÃO: A coluna escolhida precisa ter sido originalmente categórica e posteriormente encodada, as colunas numéricas não podem estar escalonadas e não pode ter havido aumento ou redução de features.')

    preprocessing9_1, preprocessing9_2, preprocessing9_3 = st.columns(3)

    with preprocessing9_1:
        smote = st.selectbox('Synthesis Target:', df2.columns)

    with preprocessing9_2:
        metodo = st.selectbox('Synthesis Method:', ['Nenhum', 'SMOTE'])

    if metodo == 'Nenhum':
        pass

    elif metodo == 'SMOTE':
        # O SMOTE procura 5 integrantes (k neighbours) daquele grupo. Portanto, se a coluna não tiver no mínimo 5 linhas daquele grupo, o SMOTE não vai funcionar. 
        df3 = df2.groupby(smote).filter(lambda x: len(x) > 5)                       # Para excluir linhas contendo grupos com menos de 5 integrantes.
        df3.reset_index(inplace=True, drop=True)

        st.markdown(f'Base de Dados Antes do SMOTE {df3.shape}')
        st.dataframe(df3)

        y = pd.DataFrame(df3[smote], columns=[smote])                               # Para garantir que y esteja no formato Pandas DataFrame e não Pandas Series.
        X = df3.drop(columns=[smote], axis=1)

        X_res, y_res = SMOTE(sampling_strategy='minority').fit_resample(X, y)

        df4 = X_res
        df4[smote] = y_res

        df5 = encoder_selecionado.inverse_transform(df4[colunas_categoricas])
        df2 = pd.DataFrame(df5, columns=colunas_categoricas)

        df2[colunas_numericas] = df4[colunas_numericas]

        st.markdown(f'Base de Dados Depois do SMOTE {df2.shape}')
        st.dataframe(df2)

        del df3, df4, df5



with preprocessing10:
    st.subheader("EDA - Análise Exploratória de Dados - Depois")

    if st.button('Gerar Relatório de EDA'):

        report = sv.analyze(df2)                                               # Use o Sweetviz para gerar o relatório.
        report.show_html("eda_depois.html", open_browser=False)                     # Salve o .html no diretório local.

        with open("eda_depois.html", "r") as f:                                     # Leia "r" o .html do diretório local.
            html_content = f.read()

        components.html(html_content, height=800, scrolling=True)                   # Mostre o .html no Streamlit.


with preprocessing11:
    st.subheader("Análise de Fluxo")

    # if st.button('Gerar Sankey'):

    df3 = df2[list(colunas_categoricas + coluna_y)]

    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    for i in range(len(colunas_categoricas) - 1):
        
        try:                # com número soma
            df4 = df3.groupby([colunas_categoricas[i], colunas_categoricas[i+1]])[coluna_y].agg('sum').reset_index()
        except:             # com classe conta
            df4 = df3.groupby([colunas_categoricas[i], colunas_categoricas[i+1]])[coluna_y].agg('count').reset_index()


        df4.columns = ['source', 'target', 'value']
        df5 = pd.concat([df5, df4], axis=0)

    df5 = df5.sort_values(by='value', ascending=False)

    # sankey = hv.Sankey(df5)
    # hv.extension('bokeh')    
    # sankey.opts(width=1500, height=900, edge_color='source', node_color='index', node_alpha=0.5, edge_alpha=0.5, label_text_font_size='18pt')
    # # plot = hv.render(sankey)
    # hv.save(sankey, 'sankey.html', backend='bokeh')
    # # output_file("sankey.html")
    # # save(plot)
    # with open('sankey.html', 'r') as f:
    #     sankey_html = f.read()
    # st.components.v1.html(sankey_html, width=1600, height=1000, scrolling=True)  




    labels = pd.concat([df5['source'], df5['target']]).unique()

    color_palette = px.colors.qualitative.Plotly                        # Plotly, D3, Set1, Dark2
    color_map = {label: color_palette[i % len(color_palette)] for i, label in enumerate(labels)}
    link_colors = df5['source'].apply(lambda x: color_map[x]).tolist()

    fig5 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=[color_map[label] for label in labels],
        ),
        link=dict(
            source=df5['source'].apply(lambda x: labels.tolist().index(x)),
            target=df5['target'].apply(lambda x: labels.tolist().index(x)),
            value=df5['value'],
            color=link_colors,
        )
    )])

    titulo = f'Soma de {coluna_y[0]}'
    fig5.update_layout(
        title=dict(
            text=titulo,
            font=dict(size=32)
        ),        
        font_size=24,
        height=1000,
    )

    st.plotly_chart(fig5, use_container_width=True, clear_figure=False)

    del df3, df4, df5




######################### explorar melhor isso https://plotly.com/python/sankey-diagram/

# e trocar por porcentage de td q sai faz todos os nos serem do mesmo tamanho e da ideia melhor de proporcao
# e separa a coluna valor das categ na primeira coisa e cola ela como chamando valor mesmo ai n da prob de duplicidade e o if sai do loop
# aqui queremos descobrir quem é o rca q n acerta nunca o preco e depois qual é a rede de contatos dele e se ele é má influencia

# trabalhar com valor sendo a contagem de tag de preco otimo y/n ou soma de 1/0
# deixar a otimização incluir uma coluna na df original.


############# já que eu tenho a tabela de origens e destinos, ver se dá p vazer uma grafo ou uma treemash: https://holoviews.org/reference/elements/bokeh/TriMesh.html 
######### um grafo permitiria identificar os nós de onde nascem as petalas para dizer q o 24 esta andando em mas companhias p ex
##### e ajudaria a ter uma ideia de clusterização por sucesso
# trocar a df por uma ja com precos a valor presente.







with st.sidebar:

    if st.button('Salvar Base de Dados Pré-Processada'):
        conn = sqlite3.connect('unip_data_science.sqlite')                              # Salve em sqlite.
        df2.to_sql('df_transformada', conn, if_exists='replace', index=False)
        conn.close()

    if st.button('Exportar Base de Dados Pré-Processada para .csv'):
        df2.to_csv('df_treino_pre-processada.csv', index=False)
