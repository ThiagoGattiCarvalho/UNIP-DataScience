import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados e Estudos Analíticos de Big Data")
st.subheader("Prof. Thiago Gatti")

home1, home2, home3 = st.tabs(['Motivação', 'Sumário Executivo', 'Quem Somos'])

with home1:
    st.subheader("Motivação")
    st.markdown("""
    \nDecidimos fazer este projeto por ...
    """)

with home2:
    st.subheader("Sumário Executivo")
    st.markdown("""
    \nO projeto ...
    """)

with home3:
    st.subheader("Quem Somos")
    
    home3_1, home3_2 = st.columns(2)
    with home3_1:
        st.image(Image.open("homem.png"), width = 200)
        st.markdown("""
            José
            \nlinkedin...
            \nFormado em...""")

    with home3_2:
        st.image(Image.open("mulher.png"), width = 200)
        st.markdown("""
            Maria
            \nlinkedin...
            \nFormada em...""")
