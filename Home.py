import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados e Estudos Analíticos de Big Data")
st.subheader("Prof. Thiago Gatti")

home1, home2, home3, home4, home5, home6, home7, home8 = st.tabs(['Programa', 'Avaliação', 'Professor', 'Notas', 'Avisos', 'Alunos: Motivação', 'Alunos: Sumário Executivo', 'Alunos: Quem Somos'])

with home1:
    st.subheader("Programa")
    st.markdown("""
        \nO curso é focado em melhoria de negócios (ao invés de maestria em TI) através da aplicação de Ciência de Dados em Big-Data.
        O conteúdo é desenvolvido usando código open-source, com python e suas bibliotecas.
        Todas as aulas terão atividade em grupo e uso de internet. A maioria dos encontros terá pouco uso de código.
    """)

    st.markdown("#### Aula1: Palestra: O Robô no Divã: Como Comportamento Afeta a IA")
    st.markdown("""Os alunos aprendem a associar perfil comportamental à Ciência de Dados para aprimorar seus códigos e aumentar suas chances de promoção.
                Além disso, será criada a base fundamental para futuras orientações ao longo do curso.""")

    st.markdown("#### Aula2: Apresentação do Curso e setup do Python")
    st.markdown("""O curso será apresentado e a infra-estrutura necessária ao uso do Python será montada.""")

    st.markdown("#### Aula3: GUI - Interface com o Usuário usando Streamlit")
    st.markdown("""Os alunos ganham auto-confiança ao gerar rapidamente o seu aplicativo em Streamlit.""")

    st.markdown("#### Aula4: Deployment no GitHub")
    st.markdown("""O aplicativo vai para internet e os estudantes passam a promovê-lo, atraindo tráfego e aumentando sua competitividade no mercado de trabalho.""")

    st.markdown("#### Aula5: Ingestão e Armazenagem de Dados em SQLite, EDA - Análise Exploratória de Dados")
    st.markdown("""Os participantes aprendem a inputar dados com o Pandas, analisá-los com o Sweetviz e armazená-los com o SQLite.""")

    st.markdown("#### Aula6: Preparação de Dados com Scikit-Learn")
    st.markdown("""Os integrantes se chocam contra a realidade e perceberm que os dados não são perfeitos. 
        Nessa aula, eles aprendem a corrigir defeitos no banco de dados para garantir que os algorítmos de Machine Learning funcionem.""")


with home6:
    st.subheader("Alunos: Motivação")
    st.markdown("""
    \nDecidimos fazer este projeto por ...
    """)

with home7:
    st.subheader("Alunos: Sumário Executivo")
    st.markdown("""
    \nO projeto ...
    """)

with home8:
    st.subheader("Alunos: Quem Somos")
    
    home8_1, home8_2 = st.columns(2)
    with home8_1:
        st.image(Image.open("homem.png"), width = 200)
        st.markdown("""
            José
            \nlinkedin...
            \nFormado em...""")

    with home8_2:
        st.image(Image.open("mulher.png"), width = 200)
        st.markdown("""
            Maria
            \nlinkedin...
            \nFormada em...""")
