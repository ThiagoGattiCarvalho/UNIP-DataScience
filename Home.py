import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados na Prática")
st.subheader("Prof. Thiago Gatti")

home1, home2 = st.tabs(['Programa', 'Professor'])

with home1:
    st.subheader("Objetivos")
    st.markdown("""
        \n1) Motivar o aluno mostrando que é fácil criar aplicativos na nuvem.
        \n2) Comparar Python, KNIME e Power BI ao longo de um projeto de Ciência de Dados.
        \n3) Pró-atividade, convertendo conhecimento em aplicações tangíveis.
        \n4) Aplicar conceitos de Gamification, com a visualização em tempo real das reações da Inteligência Artificial.
        \n5) Foco em negócio, criando motores de recomendação indicando decisões úteis, objetivas, práticas, assertivas e estratégicas.
        \n6) Melhorar a comunicação, trabalhando comportamento e incentivando a participação em projetos multi-departamentais dentro da empresa.
        \n7) Aproximar a universidade do mercado, com o chancelamento dos projetos que proporcionam melhoria efetiva nos resultados.
    """)

    st.subheader("Avaliação")
    st.markdown("""
        \nO desempenho em aula será medido através de três pilares:
        \nA) Participação: Entregando o que for pedido, sendo pontual, fazendo comentários construtivos, apresentando inovações e gerando tráfego para colegas e intituições.
        \nB) Domínio de uma tecnologia: Usando as opções No-Code e/ou criando e hospedando um aplicativo em Python com o que for ensinado em aula.
        \nC) Business Case (.pdf ou .pptx) com: 
                \n- Data Story Telling, contando a história do que acontace na empresa através de dados e gráficos,
                \n- Analytics, indo desde o descritivo até o prescritivo, passando por diagnóstico e previsões, 
                \n- Decision Automation, indicando tomadas de decisão otimizadas, estratégicas, geradas automaticamente. 
        \nOs cases aprovados serão chancelados para serem entregues a um diretor da área de negócios da empresa.
    """)

    st.subheader("Recursos")
    st.markdown("""
        \nO curso é desenvolvido com ferramentas gratuitas, como Python, KNIME e Power BI.
        Os alunos que preferirem não programar podem usar o KNIME, e os que desejarem maior autonomia usam o Python. Todos usarão o Power BI.   
        As aula terão discussões e atividades em em grupo e os alunos aprenderão a utilizar um modelo comportamental 
        para aprimorar seus aplicativos e códigos, e suas habilidades de comunicação, fomentando crescimento pessoal e profissional.
    """)

    st.subheader("Perguntas que você será capaz de responder ao término do curso")
    st.markdown("""
        \nAo longo do curso, a cada pequeno passo, você será capaz de responder a perguntas muito comuns nas empresas, como:
        \nI) Como demonstrar e maximizar ROI?
        \nII) Como definir metas de executivos?
        \nIII) Como fazer planejamento estratégico simulando cenários?
        \nIV) Qual é o preço que maximiza lucro?
        \nV) Qual é a previsão estatística de vendas com precisão acima de 95%?
        \nVI) Qual deve ser o estoque de segurança estatístico que garante 90% de nível de serviço?
        \nAgora troque as palavras: vendas, estoque, preço, ROI, etc. por quaisquer outras colunas dentro da sua base de dados e você terá uma idéia do poder que você tem nas mãos.
    """)

    st.subheader("Conteúdo")

    st.markdown("#### Módulo 1: Palestra: O Robô no Divã: Como Comportamento Afeta a IA")
    st.markdown("""Os alunos aprendem a associar comportamento à Ciência de Dados para aprimorar seus aplicativos.
                Além disso, será criada a base para as discussões ao longo do curso.""")

    st.markdown("#### Módulo 2: Setup e Interface de Usuário")
    st.markdown("""O programa de aulas será apresentado e a infra-estrutura necessária será estabelecida.
                Serão instalados o KNIME, Power BI e Python com as suas bibliotecas e Visual Studio Code.
                Os alunos ganham auto-confiança ao gerarem aplicativos em múltiplas plataformas rapidamente.""")

    st.markdown("#### Módulo 3: Análise Exploratória de Dados e Deployment")
    st.markdown("""Os alunos fazem o deployment do app na nuvem com o GitHub, ingerem os dados com Pandas, armazenam em SQLite3 e analisam com o Sweetviz.""")
    st.markdown("""O app vai para a internet e os estudantes passam a promovê-lo, atraindo tráfego e aumentando sua competitividade no mercado de trabalho.""")

    st.markdown("#### Módulo 4: Pré-Processamento")
    st.markdown("""A turma aprende a usar o Scikit-Learn para preparar os dados usando Imputers, Scalers e Encoders.""")
    st.markdown("""Os integrantes se chocam contra a realidade e percebem que os bancos de dados não são perfeitos. 
                Nessa aula, eles aprendem a corrigir defeitos no banco de dados para garantir que os algorítmos de Machine Learning funcionem.""")

    st.markdown("#### Módulo 5: Engenharia de Features")
    st.markdown("""Os participantes aprendem a usar a biblioteca Statsmodels para reduzir a quantidade de recursos da base de dados e a usar o Seaborn para gerar gráficos e dashboards.""")

    st.markdown("#### Módulo 6: Estatística")
    st.markdown("""Os integrantes aprendem como a máquina usa Estatística para avaliar e ajustar os dados, e atentam para os erros mais comuns cometidos pelos analistas. 
                Eles também aprendem a melhorar a qualidade dos seus entregáveis com dicas práticas e troca de experiências, reduzindo também o retrabalho.)
                As bibliotecas utilizadas neste módulo serão Scipy, Fitter, Statsmodels e Seaborn.""")

    st.markdown("#### Módulo 7: Machine Learning")
    st.markdown("""Os alunos aprendem os conceitos básicos de Inteligência Artificial, seus tipos, as diferenças entre os modelos, simulação de cenários e a criação de Pipelines.""")
    st.markdown("""Eles também aprendem a avaliar o output dos modelos de aprendizado de máquina para realizar ajustes e reiniciar o loop desde a preparação e a encomenda de dados com Scikit-Learn e Imbalanced-Learn.""")

    st.markdown("#### Módulo 8: Produção")  # Planejamento Estratégico e Finanças cross-validation? comparacao de modelos? simulação de cenários?

    st.markdown("#### Módulo 9: Analytics") # imbalanced-learn # shap # gestão por exceção


with home2:
    image_thiago = Image.open("Thiago.png") 
    st.image(image_thiago, width = 250)
    st.title("Thiago Gatti")
    st.subheader("Supply Chain Management & Data Science")
    st.markdown("""
    \nDiretor de Ciência de Dados na Orkideon (https://www.orkideon.com), especialista em recuperação de empresas.
    \nProfissional orientado a resultados, com bom trânsito em vários departamentos e histórico comprovado de aumento de ROI em diversos segmentos.
    \nReconhecido por intercambiar tecnologia e business e pelo expertise em Planejamento Integrado de Negócios (IBP) (ou S&OP avançado), 
        unindo Supply Chain, Vendas e Finanças com Ciência de Dados e Inteligência Artificial para otimizar a tomada de decisão e torná-la data-driven. 
    \nHábil em combinar visão estratégica com solução prática de problemas e liderança para impulsionar a inovação e elevar a gestão em organizações voltadas para o futuro.
    \nhttps://www.linkedin.com/in/thiagocscp/
    """)



# with home3:
#     st.subheader("Alunos: Motivação")
#     st.markdown("""
#     \nDecidimos fazer este projeto por ...
#     """)

# with home4:
#     st.subheader("Alunos: Sumário Executivo")
#     st.markdown("""
#     \nO projeto ...
#     """)

# with home5:
#     st.subheader("Alunos: Quem Somos")
    
#     home8_1, home8_2 = st.columns(2)
#     with home8_1:
#         st.image(Image.open("homem.png"), width = 200)
#         st.markdown("""
#             José
#             \nlinkedin...
#             \nFormado em...""")

#     with home8_2:
#         st.image(Image.open("mulher.png"), width = 200)
#         st.markdown("""
#             Maria
#             \nlinkedin...
#             \nFormada em...""")
