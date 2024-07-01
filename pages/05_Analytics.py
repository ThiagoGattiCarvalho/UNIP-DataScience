import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
from scipy.stats import rv_discrete


st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados na Prática")
st.subheader("Prof. Thiago Gatti")

try:                                                                                # Se existir uma base de dados em sql carrege.
    conn = sqlite3.connect('unip_data_science.sqlite')
    df = pd.read_sql('select * from df', conn)
    target = pd.read_sql('select * from df_target', conn)
    conn.close()
    del conn
except:                                                                             # Caso contrário pare.
    st.stop()

colunas_numericas = sorted(list(df.select_dtypes(include=['number']).columns))      # Separe as colunas numéricas das categoricas para facilitar depois.
colunas_categoricas = sorted(list(df.select_dtypes(include=['object']).columns))

st.markdown("""
## 🚧 This page is a Work in Progress 🚧
""")

# Add a construction GIF or image
st.image("https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", caption='Under Construction')


# a1, a2, a3, a4, a5, a6 = st.tabs(['Lv1 - Descritivo', 'Lv2 - Diagnóstico', 'Lv2.5 - Estratégico/Simulação de Cenários', 'Lv3 - Preditivo', 'Lv4 - Prescritivo', 'Dashboard'])

# with a1:
#     st.markdown('##### Lv1 - Descritivo')

# with a2:
#     st.markdown('##### Lv2 - Diagnóstico')
    
#     a2_1, a2_2, a2_3, a2_4 = st.columns(4)
    
#     with a2_1:

#         ic = st.slider('Intervalo de Confiança:', min_value=0.0, max_value=1.0, value=0.8, step=0.01)

#         def CalcularSeguranca(data, ic):
#             inverse_cdfs = {}
#             for column in data.columns:
#                 values, counts = np.unique(data[column], return_counts=True)
#                 probabilities = counts / len(data[column])
                
#                 # Ensure probabilities sum up to 1
#                 # total_probability = probabilities.sum()
#                 # normalized_probabilities = probabilities / total_probability
                
#                 # Calculate inverse_cdf for each column
#                 inverse_cdf = rv_discrete(values=(values, probabilities)).ppf(ic)
#                 inverse_cdf = np.round(inverse_cdf, 4)
                
#                 # Store inverse_cdf for the column
#                 inverse_cdfs[f'{column}_Segurança'] = inverse_cdf
            
#             return pd.Series(inverse_cdfs)


#     df2 = df.groupby(colunas_categoricas).apply(lambda group: CalcularSeguranca(group[colunas_numericas], ic)).reset_index()

#     st.markdown(f'##### Valores com Segurança/CDF = {ic}')
#     st.dataframe(df2)



# with a3:
#     st.markdown('##### Lv2.5 - Estratégico/Simulação de Cenários')

#     a3_1, a3_2, a3_3, a3_4 = st.columns(4)
    
#     with a3_1:
#         coluna = st.selectbox('Meta de:', colunas_numericas)

#     with a3_2:
#         meta = st.number_input('Valor:', 0.0)

#     with a3_3:
#         aumento = st.number_input('Ou multiplicar os valores acima da meta por:', 0.0)

#     # Existem grupos mais luvrativos que outros, por isso não é uma boa idéia definir uma meta de ROI igual para todos.
#     # Existem grupos com preço totalmente fora da realidade, por isso, metas de aumentar X% ROI não funcionam.
#     # Solução: Se o ROI já for acima da meta, então aumentar X%. Se for abaixo, então atingir a meta.
#     df2[f'{coluna}_Meta'] = df2[f'{coluna}_Segurança'].apply(lambda x: meta if x <= meta else x * aumento)

#     st.markdown(f'##### Metas de {coluna}')
#     st.dataframe(df2)

#     st.markdown(f'##### Cenários Sugeridos Para Forecast de {coluna}')
#     df2.columns = df2.columns.str.replace('_Segurança', '', regex=True)
#     df2.drop([coluna], inplace=True, axis=1)
#     df2.columns = df2.columns.str.replace('_Meta', '', regex=True)
#     st.dataframe(df2)
#     # Estes cenários consideram períodos passados, mas isso pode ser editado por fora.

#     if st.button('Exportar Cenários Sugeridos para .csv'):
#         df3 = df2
#         df3.drop([target.iloc[0,0]], inplace=True, axis=1)
#         df3.to_csv('df_cenarios_ai.csv', index=False)
#         del df3

#     # SUGESTÃO:
#     # Mostrar quanto cada linha difere da sua mediana e converter para dinheiro para indicar potencial de ganho.


# with a4:
#     st.markdown('##### Lv3 - Preditivo')

#     df_prever = st.file_uploader("Dataframe a prever:")
#     if df_prever is not None:
#         df2 = pd.read_csv(df_prever, encoding='iso-8859-1')

#         conn = sqlite3.connect('unip_data_science.sqlite')
#         target = pd.read_sql('select * from df_target', conn)
#         conn.close()
#         del conn

#         with open('model.pkl', 'rb') as file:
#             pipeline = pickle.load(file)    

#         forecast = pipeline.predict(df2)

#         df2[f'{target.values[0][0]}_Forecast'] = forecast


#         st.markdown(f'##### Forecast {df2.shape}')
#         st.dataframe(df2)

#         if st.button('Exportar Forecast para .csv'):
#             df2.to_csv('df_forecast.csv', index=False)


#     # Qual é a probabilidade do cenário simulado ocorrer?

# with a5:
#     st.markdown('##### Lv4 - Prescritivo')

# with a6:
#     st.markdown('##### Dashboard')
