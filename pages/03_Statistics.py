import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde, f_oneway, ttest_ind                     # Para cálculos estatísticos.
from statsmodels.stats.power import tt_solve_power, TTestPower                      # Idem
import statsmodels.api as sm                                                        # Idem.


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

colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))      # Separe as colunas numéricas das categoricas para facilitar depois.
colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

statistics1, statistics2, statistics3, statistics4, statistics5, statistics6 = st.tabs(['Distribuição Contínua x Discreta', 'PDF e CDF', 'Erro', 'ANOVA', 'Correlação e T-Test', 'Power Analysis'])


with statistics1:
    st.subheader("Distribuição Contínua x Discreta")

    statistics1_1, statistics1_2, statistics1_3, statistics1_4 = st.columns(4)

    with statistics1_1:
        coluna = st.selectbox('Coluna:', colunas_numericas)

    df = df.dropna()
    
    fig = plt.figure(figsize=(5, 3))
    
    sns.histplot(data=df,                                                           # Esse gráfico será um histograma com os dados da coluna escolhida.
                x=coluna,
                bins=10,                                                            # Segmentado em 10 barras.
                kde=True,                                                           # Transforme discreto em contínuo através de uma função.
                # alpha=0,                                                            # Se quiser ocultar as barras use 1) Transparência: alpha=0.
                # edgecolor=None,                                                     # 2) Não pinte as bordas das barras.
                )             

    st.pyplot(fig, use_container_width=False, clear_figure=False)                   # Mostre a figura no Streamlit.


with statistics2:
    st.subheader("PDF e CDF")

    statistics2_1, statistics2_2, statistics2_3, statistics2_4 = st.columns(4)

    with statistics2_1:
        min_valor = st.number_input("Valor Mínimo:", value=0.0)

    with statistics2_2:
        max_valor = st.number_input("Valor Máximo:", value=0.0)

    kde = gaussian_kde(df[coluna])                                                  # Use o KDE do gaussian_kde para criar a função contínua usando os dados da coluna selecionada.
    x_values = np.linspace(df[coluna].min(), df[coluna].max(), 100)                 # Crie só 100 valores aleatórios entre o min e o max estabelecidos para reduzir processamento.
    pdf_values = kde(x_values)                                                      # Aplique a KDE sobre os dados gerados para imitar a distribuição real.
    interval_mask = (x_values >= min_valor) & (x_values <= max_valor)
    pdf = np.sum(pdf_values[interval_mask]) / np.sum(pdf_values)

    cdf_values = np.cumsum(pdf_values) * (x_values[1] - x_values[0])                # A CDF é a integral da PDF, ou a soma acumulada das diferenças.
    min_index = np.abs(x_values - min_valor).argmin()
    max_index = np.abs(x_values - max_valor).argmin()
    cdf = cdf_values[max_index] - cdf_values[min_index]

    statistics2_coluna1, statistics2_coluna2 = st.columns(2)

    with statistics2_coluna1:
        fig = plt.figure(figsize=(4, 3))
        sns.lineplot(x=x_values,                                                    # Esse gráfico será um histograma com os dados da coluna escolhida.
                    y=pdf_values,
                    )             
        plt.fill_between(x_values, pdf_values, where=interval_mask, color='lightblue')
        plt.ylim(0)
        plt.title(f'PDF={pdf:.2f}')
        plt.xlabel(coluna)
        plt.ylabel('PDF')    
        st.pyplot(fig, use_container_width=False, clear_figure=False)               # Mostre a figura no Streamlit.

    with statistics2_coluna2:
        fig = plt.figure(figsize=(4, 3))
        sns.ecdfplot(data=df,                                                       # Esse gráfico será um histograma com os dados da coluna escolhida.
                    x=coluna,
                    )             
        plt.ylim(0)
        titulo = f'CDF={cdf:.2f}'
        plt.title(titulo)
        plt.xlabel(coluna)
        plt.ylabel('CDF')    
        plt.fill_between(x_values[min_index:max_index+1], cdf_values[min_index:max_index+1], color='lightblue')
        st.pyplot(fig, use_container_width=False, clear_figure=False)               # Mostre a figura no Streamlit.


with statistics3:
    st.subheader("Erro")

    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.set_ylabel('PDF')
    ax1.plot(x_values, pdf_values)
    plt.xlabel(coluna)
    plt.ylim(0)

    for i in range(len(x_values)):
        if interval_mask[i]:
            ax1.text(x_values[i], pdf_values[i], f'{pdf_values[i]:.2f}', ha='center', va='bottom', color='red')

    ax2 = ax1.twinx() 
    ax2.set_ylabel('Erro')  
    error_values = np.where(interval_mask, 0, pdf_values)
    ax2.plot(x_values, error_values, color='red', linestyle='--')
    plt.ylim(0)

    st.pyplot(fig, use_container_width=False, clear_figure=False)


with statistics4:
    st.subheader("ANOVA")

    statistics4_1, statistics4_2, statistics4_3, statistics4_4 = st.columns(4)

    with statistics4_1:
        eixo_x = st.selectbox('Grupo:', colunas_categoricas)

    with statistics4_2:
        cor = st.selectbox('Cor:', colunas_categoricas) 

    with statistics4_3:
        # eixo_y = st.selectbox('Eixo y ANOVA:', colunas_numericas) 
        eixo_y = coluna 

    groups = df[eixo_x].unique()
    group_data = [df[df[eixo_x] == group][eixo_y] for group in groups]
    f_statistic, p_value = f_oneway(*group_data)
    confianca = 1 - p_value

    plt.figure(figsize=(5, 3))
    sns.boxplot(data=df, x=eixo_x, y=eixo_y, hue=cor, palette='viridis')
    titulo = f'ANOVA\np-value = {p_value:.2f}, Confiança = {confianca:.2f}, Significância (Alpha) = 0.05'
    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.legend(title=cor)
    plt.legend(title=cor, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, bbox_transform=plt.gcf().transFigure)
    st.pyplot(plt.gcf(), use_container_width=False, clear_figure=False)

    if p_value <= 0.05:
        st.write(f'A diferença entre os grupos é grande. A separação entre os grupos pode ser mantida com com {confianca:.4f} de confiança.')  
    else:
        st.write('A diferença entre os grupos é baixa. Portanto, é recomendável consolidá-los em um único grupo.')  


with statistics5:

    statistics5_coluna1, statistics5_coluna2 = st.columns(2)

    with statistics5_coluna1:

        plt.figure(figsize=(5, 3))
        sns.heatmap(data=df[colunas_numericas].corr(method='spearman'), annot=True, cmap='viridis', fmt='.2f', annot_kws={"size": 6})
        titulo = f'Matriz de Correlações'
        plt.title(titulo)
        st.pyplot(plt.gcf(), use_container_width=True, clear_figure=False)

    with statistics5_coluna2:

        pvalues = pd.DataFrame(np.zeros((len(colunas_numericas), len(colunas_numericas))), columns=colunas_numericas, index=colunas_numericas)
        for col1 in colunas_numericas:
            for col2 in colunas_numericas:
                statistic, pvalue = ttest_ind(df[col1].dropna(), df[col2].dropna(), nan_policy='omit')

        plt.figure(figsize=(5, 3))
        sns.heatmap(data=pvalues, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 6})
        titulo = f'T-Test contendo P-values'
        plt.title(titulo)
        st.pyplot(plt.gcf(), use_container_width=True, clear_figure=False)


with statistics6:

    st.subheader("Power Analysis")

    n = tt_solve_power(effect_size=0.2, alpha=0.05, power=0.8, alternative='two-sided')
    st.write(f'O número mínimo de linhas por grupo (classes distintas por coluna categórica), com 80% certeza, deve ser {n:.0f}.')
    st.write('A contagem de linhas por grupo/label/classe individual em cada feature/coluna da base de dados é:')

    for col in list(df[colunas_categoricas].columns):
        contagem = df.groupby([col]).size()
        st.write(contagem.sort_values(ascending=False))



# with statistics4:
#     st.subheader("Inferência Bayesiana")

#     statistics4_1, statistics4_2, statistics4_3, statistics4_4 = st.columns(4)

#     with statistics4_1:
#         coluna = st.selectbox('Hipótese:', colunas_numericas)

#     with statistics4_2:
#         valor = st.number_input("Simulação:", value=0.0)

#     with statistics4_3:
#         min_valor = st.number_input("Valor Mínimo:", value=0.0)

#     with statistics4_4:
#         max_valor = st.number_input("Valor Máximo:", value=0.0)

#     amostra = df[coluna].dropna().values                                            # Desconsidere valores nulos para poder calcular.

#     mean = amostra.mean()
#     std = amostra.std()

#     kde = gaussian_kde(amostra)                                                     # Aplique o KDE...
#     prob_pdf = kde(valor)[0]                                                        # ... e retorne o primeiro [0] elemento entre os outputs do KDE.
#     prob_cdf = np.mean(amostra <= valor)

#     # Gere dados randômicos (só para plotar no gráfico) seguindo uma distribuição normal, mesmo sabendo que os dados não seguem uma normal.
#     x_values = np.linspace(amostra.min(), amostra.max(), 1000)                      
#     pdf_values = kde(x_values)                                                      # E plique a KDE dos dados reais sobre os dados gerados.

#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))                                  # A figura fig e seus eixos/gráficos axs serão subplots totalizando 2 linhas e 2 colunas.
#                                                                                     # Serão subplots de 1 a 3, e a figura terá o tamanho tal.
#     ax1 = axs[0, 0]                                                                 # O eixo/gráfico ax1 fica na linha 0, coluna 0.
#     ax2 = ax1.twinx()                                                               # ax2 será sobreposto twinx ao anterior. Isso é um gráfico de 2 eixos.
#     ax3 = axs[0, 1]                                                                 # ax3 ficará na linha 0, coluna 1.
#     ax4 = axs[1, 0]                                                                 # E assim por diante.
#     ax5 = axs[1,1]

#     sns.histplot(amostra,                                                           # Gere um histograma com os dados da coluna escolhida.
#                  kde=False,                                                         # Não use o KDE que vem com o Seaborn.
#                  ax=ax1,                                                            # Plote o subplot...
#                  color='skyblue',                                                   # ... na cor azul.
#                  bins=30,                                                           # Mostre 30 barras.
#                  )             

#     ax1.axvline(valor,                                                              # Plote uma linha vertical...
#                 color='r',                                                          # ... na cor vermelha 'r'...
#                 linestyle='--',                                                     # ... tracejada.
#                 label=f'Simulação = {valor}',                                       # Adicione uma etiqueta contendo o texto ...
#                 )     
    
#     ax1.set_xlabel(coluna)                                                          # Chame o eixo x de ...
#     ax1.set_ylabel('Frequência')                                                    # Chame o eixo y de ...
#     ax1.legend(loc='upper left')                                                    # Coloque a legenda em cima à esquerda.
    
#     # Dê um título. \n pula linha, f'' é conhecida como f-string, e permite inserir variáveis no meio do texto com {}.
#     titulo = f'Probabilidade de {coluna}={valor} é PDF={prob_pdf:.2f}'
#     ax1.set_title(titulo)

#     ax2.plot(x_values, pdf_values, color='blue', label='KDE')                       # Idem, porém, plote os valores gerados no eixo 2.
#     ax2.set_ylabel('PDF')
#     ax2.legend(loc='upper right')
#     ax2.set_ylim(0)                                                                 # A linha toca o eixo x no zero.
    
#     cdf_values = np.cumsum(amostra <= valor) / len(amostra)                         # Faça a média do accumulado até o valor simulado.
#     ax3.plot(np.sort(amostra),                                                      # Plote uma amostra da coluna selecionada, para salvar processamento. 
#              cdf_values,                                                            # Plote o CDF.
#              label='CDF', 
#              color='blue',
#              )
    
#     ax3.axvline(valor, color='r', linestyle='--', label=f'Simulação = {valor}')     # Adicione uma linha vertical para o valor simulado.
#     ax3.set_xlabel(coluna)
#     ax3.set_ylabel('CDF')
#     ax3.legend()                                                                    # Mostre legenda onde achar melhor.
#     titulo = f'Probabilidade de {coluna}<={valor} é CDF={prob_cdf:.2f}'
#     ax3.set_title(titulo)
#     ax3.set_ylim(0)


#     interval_mask = (x_values >= min_valor) & (x_values <= max_valor)
#     interval_prob_pdf = np.sum(pdf_values[interval_mask]) / np.sum(pdf_values)
#     ax4.plot(x_values, pdf_values, color='blue', label='KDE')
#     ax4.fill_between(x_values, pdf_values, where=interval_mask, color='lightblue', alpha=0.5)
#     ax4.set_xlabel(coluna)
#     ax4.set_ylabel('PDF')
#     ax4.legend()
#     titulo = f'Probabilidade de {min_valor}<={coluna}<={max_valor} é PDF={interval_prob_pdf:.2f}'
#     ax4.set_title(titulo)
#     ax4.set_ylim(0)

#     for ax in axs.flatten():                                                        # Escreva os valores do eixo x na vertical.
#         plt.sca(ax)
#         plt.xticks(rotation=90)

#     ax5.axis('off')                                                                 # Não mostre nada na linha 1, coluna 1.

#     plt.subplots_adjust(wspace=0.3, hspace=0.6)

#     st.pyplot(fig, use_container_width=False, clear_figure=False)                   # E mostre a figura toda completa no Streamlit.

#     prob_pdf = norm.pdf(valor, mean, std)                                           # Para comparação, calcule a PDF como se fosse uma Normal.
#     st.markdown(f'Se, a distribuição fosse normal, a PDF seria {prob_pdf:.2f}')
