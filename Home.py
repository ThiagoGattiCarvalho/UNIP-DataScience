import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados e Estudos Analíticos de Big Data")
st.subheader("Prof. Thiago Gatti")

home1, home2, home3, home4, home5, home6, home7, home8 = st.tabs(['Programa', 'Professor', 'Avaliação', 'Notas', 'Avisos', 'Alunos: Motivação', 'Alunos: Sumário Executivo', 'Alunos: Quem Somos'])

with home1:
    st.subheader("Programa")
    st.markdown("""
        \nO curso tem 2 objetivos:
        \n1) Motivar o aluno mostrando que fácil criar um aplicativo na nuvem, como este, para se auto-promover.
        \n2) Melhorar o seu negócio com análises corretas e tomadas de decisão otimizadas, advindas da Ciência de Dados com Machine Learning.
        \nO desempenho dos alunos será medido através da sua participação em aula, criação do aplicativo e melhora efetiva do negócio, explicitada através de um business case destinado a profissionais de business (ao invés de TI).
        \nO conteúdo do curso é desenvolvido usando código open-source, com Python e suas bibliotecas.
        As aulas terão atividades em grupo, uso de internet e pouco código.
        \nOs alunos também aprendem um modelo Junguiano de perfil comportamental e, em seguida, são incentivados a aplicar o modelo compartilhando e comentando os trabalhos dos colegas
        para aprimorar as suas habilidades de comunicação e para construir raport, contribuindo com seu crescimento pessoal e profissional.
    """)

    st.markdown("#### Aula1: Palestra: O Robô no Divã: Como Comportamento Afeta a IA")
    st.markdown("""Os alunos aprendem a associar um perfil comportamental Junguiano à Ciência de Dados para aprimorar seus códigos e aumentar suas chances de promoção na empresa.
                Além disso, será criada a base para as discussões ao longo do curso.""")

    st.markdown("#### Aula2: Apresentação do Curso, setup do Python e criação da GUI - interface com o usuário")
    st.markdown("""O programa de aulas será apresentado e a infra-estrutura necessária ao uso do Python será montada, bem como a interface com o usuário.
                 O código será feito com Visual Studio Code (ao invés de Jupyter) e os alunos ganham auto-confiança ao gerar rapidamente um aplicativo em Streamlit.""")

    st.markdown("#### Aula3: Deployment")
    st.markdown("""Os alunos fazem o deployment do app na nuvem com o GitHub, ingerem os dados com Pandas, armazenam em SQLite3 e analisam com o Sweetviz.""")
    st.markdown("""O app vai para internet e os estudantes passam a promovê-lo, atraindo tráfego e aumentando sua competitividade no mercado de trabalho.""")

    st.markdown("#### Aula4: Pré-Processamento de Dados")
    st.markdown("""A turma aprende a usar o Scikit-Learn para preparar os dados usando Imputers, Scalers e Encoders.""")
    st.markdown("""Os integrantes se chocam contra a realidade e perceberm que os bancos de dados não são perfeitos. 
                Nessa aula, eles aprendem a corrigir defeitos no banco de dados para garantir que os algorítmos de Machine Learning funcionem.""")

    st.markdown("#### Aula5: Engenharia de Features")
    st.markdown("""Os participantes aprendem a usar mais duas bibliotecas: Statsmodels e Seaborn para aumentar e reduzir o número de recursos e, em seguida, visualizar as análises graficamente através da criação de dashboards.""")

    st.markdown("#### Aula6: Machine Learning")
    st.markdown("""Os alunos aprendem os conceitos básicos de Inteligência Artificial, as diferenças entre os modelos, a simulação de cenários e a criação de Pipelines.""")
    st.markdown("""Eles também aprendem a avaliar o output dos modelos de aprendizado de máquina e a realizar ajustes.""")

    st.markdown("#### Aula7: Estatística para Machine Learning")
    st.markdown("""Os integrantes aprendem o que está por trás dos métodos de avaliação da AI e os erros mais comuns dos analistas atualmente para não repetirem os mesmos erros e para reduzirem o retrabalho com Ciência de Dados.""")

    st.markdown("#### Aula8: TBD")

    st.markdown("#### Aula9: TBD")


with home2:
    image_thiago = Image.open("Thiago.png") 
    st.image(image_thiago, width = 250)
    st.title("Thiago Gatti")
    st.subheader("Supply Chain Management & Data Science")
    st.markdown("""
    \nProfissional orientado a resultados, com bom trânsito entre áreas e histórico comprovado de aumento de ROI em diversos setores.
    \nReconhecido pela facilidade de transitar entre tecnologia e business e pelo expertise em Planejamento Integrado de Negócios (IBP) (ou S&OP avançado), unindo fortemente Supply Chain, Vendas e Finanças, e aplicando Ciência de Dados e Inteligência Artificial para otimizar a tomada de decisão e torná-la data-driven. 
    \nHábil em combinar visão estratégica com solução prática de problemas e liderança para impulsionar a inovação e elevar a gestão em organizações voltadas para o futuro.
    \nhttps://www.linkedin.com/in/thiagocscp/
    """)



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
