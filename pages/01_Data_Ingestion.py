import streamlit as st                                                              # Instalada na aula passada.
import pandas as pd                                                                 # Default do Streamlit
import sqlite3                                                                      # Default do Python
import sweetviz as sv                                                               # Instalada agora
import streamlit.components.v1 as components                                        # Instalada agora

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

ingestion1, ingestion2 = st.tabs(['Ingestão de Dados', 'EDA - Análise Exploratória de Dados - Antes'])  # 2 guias.

with ingestion1:
    st.subheader("Ingestão de Dados")

    df_upload = st.file_uploader("Dataframe para importar:")                        # Uma variável será o arquivo importado
    if df_upload is not None:

        try:                                                                        # Importe csv ou Excel.
            df = pd.read_csv(df_upload, encoding='iso-8859-1')
        except:
            df = pd.read_excel(df_upload, index_col=None, engine='openpyxl')

        conn = sqlite3.connect('unip_data_science.sqlite')                          # Salve em sqlite.
        df.to_sql('df', conn, if_exists='replace', index=False)
        conn.close()
        del df, df_upload, conn                                                     # Limpe a memória para agilizar o app.

        st.title("Pressione F5!")                                                   # Atualize o cache.

    try:                                                                            # Mostre o que existe.
        st.write(f"""##### Dataframe {df.shape})""")                                # .shape mostra quantas linhas e colunas.
        st.dataframe(df)
    except:                                                                         # Ou pare tudo e espere.
        st.stop()


with ingestion2:
    st.subheader("EDA - Análise Exploratória de Dados - Antes")

    report = sv.analyze(df)                                                         # Use o Sweetviz para gerar o relatório.
    report.show_html("eda_antes.html", open_browser=False)                                # Salve o .html no diretório local.

    with open("eda_antes.html", "r") as f:                                                # Leia "r" o .html do diretório local.
        html_content = f.read()

    components.html(html_content, height=800, scrolling=True)                       # Mostre o .html no Streamlit.