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

st.title("Ciência de Dados na Prática")
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

    coluna = st.selectbox('Analisar:', colunas_numericas, index=8)
    eixo_x = st.selectbox('por:', list(df.columns), index=3)
    item = st.selectbox('Escolher:', pd.unique(df[eixo_x]))

    filtered_df2 = df[df[eixo_x] == item]
    filtered_df = filtered_df2.dropna()
    del filtered_df2


statistics1, statistics2, statistics3, statistics4, statistics5, statistics6 = st.tabs(['Distribuição', 'Probabilidade', 'Risco, Segurança e Sobrevivência', 'ANOVA', 'Correlação', 'Potência'])


with statistics1:
    st.subheader("Distribuição")

    st.markdown(f'Qual é a diferença entre a distribuição discreta e contínua para {item}?')


    fig1 = plt.figure(figsize=(5, 3))
    sns.histplot(data=filtered_df,                                                           # Esse gráfico será um histograma com os dados da coluna escolhida.
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

    kde = gaussian_kde(filtered_df[coluna])                                                  # Use o KDE do gaussian_kde para criar a função contínua usando os dados da coluna selecionada.
    x_values = np.linspace(filtered_df[coluna].min(), filtered_df[coluna].max(), 100)                 # Crie só 100 valores aleatórios entre o min e o max estabelecidos para reduzir processamento.
    pdf_values = kde(x_values)                                                      # Aplique a KDE sobre os dados gerados para imitar a distribuição real.
    interval_mask = (x_values >= min_valor) & (x_values <= max_valor)
    pdf = np.sum(pdf_values[interval_mask]) / np.sum(pdf_values)

    # Para calcular a CDF entre os pontos
    # cdf_values = np.cumsum(pdf_values) * (x_values[1] - x_values[0])                # A CDF é a integral da PDF, ou a soma acumulada das diferenças.
    # min_index = np.abs(x_values - min_valor).argmin()
    # max_index = np.abs(x_values - max_valor).argmin()
    # cdf = cdf_values[max_index] - cdf_values[min_index]

    # Para calcular a CDF acumulada até o maior valor escolhido. Usar esta para calcular valores de segurança.
    ecdf = ECDF(filtered_df[coluna].values)
    cdf = ecdf(max_valor)

    statistics2_coluna1, statistics2_coluna2 = st.columns(2)

    with statistics2_coluna1:

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

        st.set_option('deprecation.showPyplotGlobalUse', False)
        fitter = Fitter(filtered_df[coluna], distributions=get_common_distributions())
        fitter.fit()
        fitter.plot_pdf()
        plt.title('Inferência Estatística')
        st.pyplot(use_container_width=True, clear_figure=False)                   # Mostre a figura no Streamlit.


    with statistics2_coluna2:

        fig3 = plt.figure(figsize=(4, 3))
        sns.ecdfplot(data=filtered_df,                                                       # Esse gráfico será um histograma com os dados da coluna escolhida.
                    x=coluna,
                    label='CDF',
                    )             
        plt.ylim(0)
        # plt.xlim(0)
        titulo = f'CDF={cdf:.2f}'
        plt.title(titulo)
        plt.xlabel(coluna)
        plt.ylabel('Probabilidade Acumulada (CDF)')    
        plt.fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x <= max_valor), color='lightblue')
        plt.legend()
        st.pyplot(fig3, use_container_width=True, clear_figure=False)               # Mostre a figura no Streamlit.

        st.markdown(f'Qual é a probabilidade (Intervalo de Confiança) de {coluna} estar entre {min_valor} e {max_valor} inclusive? R: {pdf:.2f}.')
        st.markdown(f'Qual é a probabilidade (Intervalo de Confiança) para {coluna} ser menor ou igual a {max_valor}? R: {cdf:.2f}.')

    st.write("Summary of best fits:")
    st.write(fitter.summary())


with statistics3:
    st.subheader("Risco, Segurança e Sobrevivência")

    statistics3_1, statistics3_2, statistics3_3, statistics3_4 = st.columns(4)

    with statistics3_1:
        ic = st.slider('Intervalo de Confiança da CDF:', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

    # mediana = np.median(filtered_df[coluna])
    # mad = median_abs_deviation(filtered_df[coluna])
    # z_robusto = (filtered_df[coluna] - mediana) / mad
    # x = z_robusto * mad + mediana

    values, counts = np.unique(filtered_df[coluna], return_counts=True)
    probabilities = counts / len(filtered_df[coluna])
        
    safety_inverse_cdf = rv_discrete(values=(values, probabilities)).ppf(ic)               # Segurança        
    safety_inverse_cdf = np.round(safety_inverse_cdf, 2)                                  
    survival_inverse_cdf = rv_discrete(values=(values, probabilities)).ppf(1 - ic)       # Sobrevivência
    survival_inverse_cdf = np.round(survival_inverse_cdf, 2)

    df_sorted = filtered_df.sort_values(by=coluna)
    values_sorted = df_sorted[coluna].values
    unique_values, unique_counts = np.unique(values_sorted, return_counts=True)
    cumulative_counts = np.cumsum(unique_counts)
    survival_function = 1 - (cumulative_counts / len(df_sorted))
    hazard_function = unique_counts / (len(df_sorted) - cumulative_counts + unique_counts)
    cumulative_hazard_function = np.cumsum(hazard_function)
    x_value_at_08_survival = unique_values[np.argmin(np.abs(survival_function - 0.8))]

    fig4, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

    sns.ecdfplot(data=filtered_df, x=coluna, ax=axes[0, 0])             
    axes[0, 0].set_ylim(0)
    titulo = f'Segurança = {ic:.2f}'
    axes[0, 0].set_title(titulo)
    axes[0, 0].set_xlabel(coluna)
    axes[0, 0].set_ylabel('Probabilidade Acumulada (CDF)')    
    axes[0, 0].fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x <= safety_inverse_cdf), color='lightblue')
    label = f'{coluna} = {safety_inverse_cdf}'
    axes[0, 0].axvline(safety_inverse_cdf, label=label, color='r', linestyle='--')
    axes[0, 0].legend()

    sns.ecdfplot(data=filtered_df, x=coluna, ax=axes[0, 1])             
    axes[0, 1].set_ylim(0)
    titulo = f'Sobrevivência = {ic:.2f}'
    axes[0, 1].set_title(titulo)
    axes[0, 1].set_xlabel(coluna)
    axes[0, 1].set_ylabel('Probabilidade Acumulada (CDF)')    
    axes[0, 1].fill_between(ecdf.x, 0, ecdf.y, where=(ecdf.x >= survival_inverse_cdf), color='lightblue')
    label = f'{coluna} = {survival_inverse_cdf}'
    axes[0, 1].axvline(survival_inverse_cdf, label=label, color='r', linestyle='--')
    axes[0, 1].legend()

    sns.lineplot(x=unique_values, y=hazard_function, drawstyle='steps-post', ax=axes[1, 0])
    axes[1, 0].set_ylim(0)
    titulo = 'Função Risco Instantâneo'
    axes[1, 0].set_title(titulo)
    axes[1, 0].set_xlabel(coluna)
    axes[1, 0].set_ylabel('Risco (de não sobreviver)')

    sns.lineplot(x=unique_values, y=survival_function, drawstyle='steps-post', ax=axes[1, 1])
    axes[1, 1].set_ylim(0)
    titulo = 'Função Sobrevivência'
    axes[1, 1].set_title(titulo)
    axes[1, 1].set_xlabel(coluna)
    axes[1, 1].set_ylabel('Probabilidade de Sobrevivência')
    axes[1, 1].fill_between(unique_values, 0.8, 1, color='lightblue')
    label = f'{coluna} 80% Sobrevivência = {x_value_at_08_survival}'
    axes[1, 1].axhline(0.8, label=label, color='r', linestyle='--')
    axes[1, 1].legend()

    plt.tight_layout()
        
    st.pyplot(fig4, use_container_width=True, clear_figure=False)

    st.markdown(f'Qual é {coluna} máximo com {ic} segurança (Intervalo de Confiança)? R: {safety_inverse_cdf}.')
    st.markdown(f'Qual é {coluna} mínimo que garante a sobrevivência com {ic} de certeza (Intervalo de Confiança)? R: {survival_inverse_cdf}.')
    st.markdown(f'Qual é {coluna} que garante 80% Sobrevivência? R: {x_value_at_08_survival}.')






with statistics4:
    st.subheader("ANOVA")

    statistics4_1, statistics4_2, statistics4_3, statistics4_4 = st.columns(4)

    with statistics4_1:
        cor = st.selectbox('Cor:', list(df.columns), index=3) 

    with statistics4_1:
        eixo_y = coluna 

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

    st.markdown(f'É possível acreditar que os grupos dentro de {coluna} são distintos?')

    if p_value <= 0.05:
        st.write(f'R: Sim, os grupos são distintos, com {confianca:.4f} de confiança.')  
    else:
        st.write('R: Não, os grupos não são distintos. Recomenda-se consolidar alguns grupos e/ou sintetizar dados.')  


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


