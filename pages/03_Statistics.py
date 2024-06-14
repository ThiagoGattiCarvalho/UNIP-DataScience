import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, median_abs_deviation, f_oneway, ttest_ind, rv_discrete, rv_continuous, norm  # Para cálculos estatísticos.
from statsmodels.stats.power import TTestIndPower                                       # Idem
from statsmodels.distributions.empirical_distribution import ECDF                       # Idem
from fitter import Fitter, get_common_distributions, get_distributions                  # Para identificar o tipo de distribuição.

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados e Estudos Analíticos de Big Data")
st.subheader("Prof. Thiago Gatti")

with st.sidebar:
    radio = st.radio('Usar base de dados:',['Original', 'Transformada'])

    if radio == 'Original':
        conn = sqlite3.connect('unip_data_science.sqlite')
        df = pd.read_sql('select * from df', conn)
        conn.close()
    elif radio == 'Transformada':
        conn = sqlite3.connect('unip_data_science.sqlite')
        df = pd.read_sql('select * from df_transformada', conn)
        conn.close()

    colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))      # Separe as colunas numéricas das categoricas para facilitar depois.
    colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

    coluna = st.selectbox('Analisar:', colunas_numericas)
    eixo_x = st.selectbox('por Grupo:', list(df.columns))


statistics1, statistics2, statistics3, statistics4, statistics5, statistics6 = st.tabs(['Distribuição', 'Probabilidade', 'Segurança e Sobrevivência', 'ANOVA', 'Correlação', 'Potência'])


with statistics1:
    st.subheader("Distribuição")

    st.markdown(f'Qual é a diferença entre a distribuição discreta e contínua para {coluna}?')

    df = df.dropna()

    fig1 = plt.figure(figsize=(5, 3))
    sns.histplot(data=df,                                                           # Esse gráfico será um histograma com os dados da coluna escolhida.
                x=coluna,
                bins=10,                                                            # Segmentado em 10 barras.
                kde=True,                                                           # Transforme discreto em contínuo através de uma função.
                label='Discreta',
                # alpha=0,                                                            # Se quiser ocultar as barras use 1) Transparência: alpha=0.
                # edgecolor=None,                                                     # 2) Não pinte as bordas das barras.
                )
    plt.title('Distribuições Contínua e Discreta')
    plt.plot([], [], label='Contínua')
    plt.legend()

    st.pyplot(fig1, use_container_width=True, clear_figure=False)                   # Mostre a figura no Streamlit.



with statistics2:
    st.subheader("Probabilidade")

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

    # Para calcular a CDF entre os pontos
    # cdf_values = np.cumsum(pdf_values) * (x_values[1] - x_values[0])                # A CDF é a integral da PDF, ou a soma acumulada das diferenças.
    # min_index = np.abs(x_values - min_valor).argmin()
    # max_index = np.abs(x_values - max_valor).argmin()
    # cdf = cdf_values[max_index] - cdf_values[min_index]

    # Para calcular a CDF acumulada até o maior valor escolhido. Usar esta para calcular valores de segurança.
    ecdf = ECDF(df[coluna].values)
    cdf = ecdf(max_valor)


    statistics2_coluna1, statistics2_coluna2 = st.columns(2)

    with statistics2_coluna1:

        st.markdown(f'Qual é a probabilidade (Intervalo de Confiança) de {coluna} estar entre {min_valor} e {max_valor} inclusive?')

        fig2 = plt.figure(figsize=(4, 3))
        sns.lineplot(x=x_values,                                                    # Esse gráfico será um histograma com os dados da coluna escolhida.
                    y=pdf_values,
                    label='PDF',
                    )             
        plt.fill_between(x_values, pdf_values, where=interval_mask, color='lightblue')
        plt.ylim(0)
        # plt.xlim(0)
        plt.title(f'PDF={pdf:.2f}')
        plt.xlabel(coluna)
        plt.ylabel('PDF - Densidade de Probabilidade')  
        st.pyplot(fig2, use_container_width=True, clear_figure=False)               # Mostre a figura no Streamlit.

        st.markdown(f'A probabilidade (Intervalo de Confiança) de {coluna} estar entre {min_valor} e {max_valor} inclusive é {pdf:.2f}.')


        st.set_option('deprecation.showPyplotGlobalUse', False)
        fitter = Fitter(df[coluna], distributions=get_common_distributions())
        fitter.fit()
        fitter.plot_pdf()
        plt.title('Inferência Estatística')
        st.pyplot(use_container_width=True, clear_figure=False)                   # Mostre a figura no Streamlit.
        
        st.write("Summary of best fits:")
        st.write(fitter.summary())



    with statistics2_coluna2:

        st.markdown(f'Qual é a probabilidade (Intervalo de Confiança) para {coluna} ser menor ou igual a {max_valor}?')

        fig3 = plt.figure(figsize=(4, 3))
        sns.ecdfplot(data=df,                                                       # Esse gráfico será um histograma com os dados da coluna escolhida.
                    x=coluna,
                    label='CDF',
                    )             
        plt.ylim(0)
        # plt.xlim(0)
        titulo = f'CDF={cdf:.2f}'
        plt.title(titulo)
        plt.xlabel(coluna)
        plt.ylabel('CDF - Acúmulo de Probabilidade')    
        plt.fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x <= max_valor), color='lightblue')
        plt.legend()
        st.pyplot(fig3, use_container_width=True, clear_figure=False)               # Mostre a figura no Streamlit.

        st.markdown(f'A probabilidade de {coluna} ser menor ou igual a {max_valor} é de {cdf:.2f} (Intervalo de Confiança).')



with statistics3:
    st.subheader("Segurança e Sobrevivência")

    statistics3_1, statistics3_2, statistics3_3, statistics3_4 = st.columns(4)

    with statistics3_1:
        ic = st.slider('Intervalo de Confiança da CDF:', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    # mediana = np.median(df[coluna])
    # mad = median_abs_deviation(df[coluna])
    # z_robusto = (df[coluna] - mediana) / mad
    # x = z_robusto * mad + mediana

    values, counts = np.unique(df[coluna], return_counts=True)
    probabilities = counts / len(df[coluna])
    inverse_cdf = rv_discrete(values=(values, probabilities)).ppf(ic)
    inverse_cdf = np.round(inverse_cdf, 2)

    statistics3_coluna1, statistics3_coluna2 = st.columns(2)

    with statistics3_coluna1:

        st.markdown(f'Qual é o(a) {coluna} máximo que garante {ic} de segurança (Intervalo de Confiança)?')

        fig4 = plt.figure(figsize=(4, 3))
        sns.ecdfplot(data=df, x=coluna)             
        plt.ylim(0)
        # plt.xlim(0)
        titulo = f'Segurança = CDF = {ic:.2f}'
        plt.title(titulo)
        plt.xlabel(coluna)
        plt.ylabel('CDF - Acúmulo de Probabilidade')    
        plt.fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x <= inverse_cdf), color='lightblue')
        label = f'{coluna} = {inverse_cdf}'
        plt.axvline(inverse_cdf, label=label, color='r', linestyle='--')
        plt.legend()
        st.pyplot(fig4, use_container_width=True, clear_figure=False)               # Mostre a figura no Streamlit.

        st.markdown(f'O(a) {coluna} máximo que garante {ic} segurança (Intervalo de Confiança) é {inverse_cdf}.')


    with statistics3_coluna2:

        st.markdown(f'Qual é o(a) {coluna} mínimo que garante a sobrevivência com {ic} de certeza (Intervalo de Confiança)?')

        inverse_cdf = rv_discrete(values=(values, probabilities)).ppf(1 - ic)
        inverse_cdf = np.round(inverse_cdf, 2)

        fig5 = plt.figure(figsize=(4, 3))
        sns.ecdfplot(data=df, x=coluna)             
        plt.ylim(0)
        # plt.xlim(0)
        titulo = f'Sobrevivência = 1 - CDF = {ic:.2f}'
        plt.title(titulo)
        plt.xlabel(coluna)
        plt.ylabel('CDF - Acúmulo de Probabilidade')    
        plt.fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x >= inverse_cdf), color='lightblue')
        label = f'{coluna} = {inverse_cdf}'
        plt.axvline(inverse_cdf, label=label, color='r', linestyle='--')
        plt.legend()
        st.pyplot(fig5, use_container_width=True, clear_figure=False)               # Mostre a figura no Streamlit.

        st.markdown(f'O(a) {coluna} mínimo que garante a sobrevivência com {ic} de certeza (Intervalo de Confiança) é {inverse_cdf}.')



with statistics4:
    st.subheader("ANOVA")

    statistics4_1, statistics4_2, statistics4_3, statistics4_4 = st.columns(4)

    with statistics4_1:
        cor = st.selectbox('Cor:', list(df.columns)) 

    with statistics4_1:
        # eixo_y = st.selectbox('Eixo y ANOVA:', colunas_numericas) 
        eixo_y = coluna 

    st.markdown(f'É possível acreditar que os grupos dentro de {coluna} são distintos?')

    groups = df[eixo_x].unique()
    group_data = [df[df[eixo_x] == group][eixo_y] for group in groups]
    f_statistic, p_value = f_oneway(*group_data)
    confianca = 1 - p_value

    fig6 = plt.figure(figsize=(5, 3))
    sns.boxplot(data=df, x=eixo_x, y=eixo_y, hue=cor, palette='viridis')
    titulo = f'ANOVA'
    plt.title(titulo)
    plt.xlabel(eixo_x)
    plt.ylabel(eixo_y)
    plt.legend(title=cor)
    plt.legend(title=cor, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, bbox_transform=plt.gcf().transFigure)
    subtitulo = f'p-value = {p_value:.2f}, Confiança = {confianca:.2f}, Significância (Alpha) = 0.05'
    plt.text(0.5, 0.96, subtitulo, ha='center', fontsize=8, transform=plt.gca().transAxes)
    st.pyplot(fig6, use_container_width=True, clear_figure=False)

    if p_value <= 0.05:
        st.write(f'Sim, os grupos são distintos, com {confianca:.4f} de confiança.')  
    else:
        st.write('Não, os grupos não são distintos. Recomenda-se consolidar alguns grupos.')  


with statistics5:

    statistics5_coluna1, statistics5_coluna2 = st.columns(2)

    with statistics5_coluna1:

        fig7 = plt.figure(figsize=(5, 3))
        sns.heatmap(data=df[colunas_numericas].corr(method='spearman'), annot=True, cmap='viridis', fmt='.2f', annot_kws={"size": 6})
        titulo = f'Matriz de Correlações'
        plt.title(titulo)
        st.pyplot(fig7, use_container_width=True, clear_figure=False)

    with statistics5_coluna2:

        pvalues = pd.DataFrame(np.zeros((len(colunas_numericas), len(colunas_numericas))), columns=colunas_numericas, index=colunas_numericas)
        for col1 in colunas_numericas:
            for col2 in colunas_numericas:
                statistic, pvalue = ttest_ind(df[col1].dropna(), df[col2].dropna(), nan_policy='omit')

        fig8 = plt.figure(figsize=(5, 3))
        sns.heatmap(data=pvalues, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 6})
        titulo = f'T-Test contendo P-values'
        plt.title(titulo)
        st.pyplot(fig8, use_container_width=True, clear_figure=False)

        st.markdown("Quando o p-value é menor que 0.05 (Significância), pode-se confiar na correlação.")

    correlacoes = {
        'Coeficiente de Correlação': ['0', '-0.1/+0.1', '-0.3/+0.3', '-0.5/+0.5', '-1/+1'],
        'Força da Relação Entre as Variáveis': ['Nula', 'Fraca', 'Moderada', 'Forte', 'Perfeita'],
    }
    df_correlacoes = pd.DataFrame(correlacoes)
    st.dataframe(df_correlacoes)

with statistics6:

    st.subheader("Potência")

    statistics6_1, statistics6_2, statistics6_3, statistics6_4 = st.columns(4)

    grupos = df[eixo_x].unique()
    contagem = len(grupos)
    power_matrix = np.zeros((contagem, contagem))
    effect_matrix = np.zeros((contagem, contagem))
    numero_ideal_de_amostras_dic = {}

    # if st.button("Realizar Teste de Potência"):

    for i in range(contagem):
        grupo1_valores = df[df[eixo_x] == grupos[i]][coluna]
        nobs1 = len(grupo1_valores)
        numero_ideal_de_amostras = TTestIndPower().solve_power(effect_size=0.2, alpha=0.05, power=0.8)
        numero_ideal_de_amostras_dic[grupos[i]] = numero_ideal_de_amostras

        for j in range(contagem):
            grupo2_valores = df[df[eixo_x] == grupos[j]][coluna]
            nobs2 = len(grupo2_valores)
            effect_size = (grupo1_valores.mean() - grupo2_valores.mean()) / np.sqrt(((nobs1-1)*grupo1_valores.var() + (nobs2-1)*grupo2_valores.var()) / (nobs1 + nobs2 - 2))
            power = TTestIndPower().power(effect_size=effect_size, nobs1=nobs1, alpha=0.05, ratio=nobs2/nobs1)
            power_matrix[i, j] = power
            effect_matrix[i, j] = effect_size


    st.markdown(f'##### Matriz de Potência para grupos de {eixo_x} considerando {coluna} (Percentual de certeza na diferença entre grupos, ou probabilidade de corretamente rejeitar a hipótese nula) (desejável > 0.8)')
    st.write(pd.DataFrame(power_matrix, index=grupos, columns=grupos).applymap("{:.2f}".format))

    st.write("##### Magnitude/Efeito da Diferença entre Grupos (d) (0.8=Alto, 0.5=Médio, 0.2=Baixo)")
    st.write(pd.DataFrame(effect_matrix, index=grupos, columns=grupos).applymap("{:.2f}".format))        

    st.write("##### Dimensionamento do Banco de Dados")
    df_ideal = pd.DataFrame(numero_ideal_de_amostras_dic, index=['Tamanho Ideal da Amostra (potência=0.8, d=0.2, alpha=0.05)']).transpose().applymap("{:.2f}".format)
    df_real = df.groupby(eixo_x).size().rename('Tamanho Real da Amostra').to_frame().applymap("{:.0f}".format)
    st.write(pd.concat([df_ideal, df_real], axis=1))


    st.write(f'IMPORTANTE: Somente considerar a análise de potência quando p-value ≤ α')


