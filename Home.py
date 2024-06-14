import streamlit as st

st.set_page_config(layout="wide")

from PIL import Image
logo = Image.open("logo.png") 
st.image(logo, width = 400)

st.title("Ciência de Dados na Prática")
st.subheader("Prof. Thiago Gatti")

home1, home2, home3, home4, home5 = st.tabs(['Programa', 'Professor', 'Alunos: Motivação', 'Alunos: Sumário Executivo', 'Alunos: Quem Somos'])

with home1:
    st.subheader("Objetivo")
    st.markdown("""
        \nO curso tem cinco objetivos:
        \n1) Motivar o aluno mostrando que é fácil criar um aplicativo na nuvem, igual a este, para se auto-promover.
        \n2) Reforçar o aprendizado aplicando conceitos de Gamification, mostrando visualmente, em tempo real, o que acontece a cada etapa do processo.
        \n3) Melhorar negócio e carreira, com análises corretas e tomadas de decisão práticas e assertivas, sob orientação de um profissional com mais de vinte anos de experiência em consultoria.
        \n4) Aprimorar a comunicação, trabalhando linguagem de negócios para elevar a eficiência do participante em projetos multi-departamentais dentro da empresa.
        \n5) Aproximar a universidade da empresa, com o chancelamento dos projetos dos alunos, claramente otimizando os negócios com cases e ferramentas práticas.
    """)

    st.subheader("Medição de Desempenho")
    st.markdown("""
        \nO desempenho em aula será medido através de três pilares:
        \nA) Participação: Entregando semanalmente o que for pedido, agregando comentários construtivos durante os fórums de discussão e gerando tráfego para os colegas e intituições.
        \nB) Aplicativo de Ciência de Dados: Efetivamente criando e hospedando um aplicativo com as funções ensinadas em aula.
        \nC) Business Case com: 
                \n- Data Story Telling, contando a história do que acontace na empresa através de dados,
                \n- Analytics, indo desde o descritivo até o prescritivo, passando por diagnóstico e previsões, 
                \n- Decision Automation, indicando tomadas de decisão otimizadas, geradas automaticamente, que possibilitam
                    fazer o planejamento estratégico,                   
                    monitorar o negócio e ajustar os planos em tempo real,
                    definir metas de executiivos,
                    detectar fraudes,
                    realizar a gestão por exceção e mais.
        \nOs cases aprovados serão chancelados para serem entregues ao um diretor da área de negócios da empresa.
""")

    st.subheader("Recursos")
    st.markdown("""
        \nO conteúdo do curso é desenvolvido usando código open-source, com Python e suas bibliotecas.
        As aulas terão atividades em grupo, uso de internet e desenvolvimento de código (fácil).
        \nOs alunos também aprendem um modelo Junguiano de perfil comportamental e, em seguida, são incentivados a aplicar o modelo compartilhando e comentando os trabalhos dos colegas
        para aprimorar as suas habilidades de comunicação e para construir rapport, contribuindo para seu crescimento pessoal e profissional.
    """)

    st.subheader("Perguntas que você será capaz de responder ao término do curso")
    st.markdown("""
        \nAo longo do curso, a cada pequeno passo, você será capaz de responder a perguntas muito comuns nas empresas, como:
        \nI) Qual é o preço que maximiza lucro?
        \nII) Qual é a previsão estatística de vendas com precisão acima de 95%?
        \nIII) Qual deve ser o estoque de segurança estatístico que garante 90% de nível de serviço?
        \nIV) Como calcular e otimizar o ROI?
        \nV) Como fazer planejamento estratégico simulando cenários?
        \nVI) Como definir metas de executivos?
        \nAgora troque as palavras: vendas, estoque, preço, ROI, etc. por quaisquer outras colunas dentro da sua base de dados e você terá uma idéia do poder que você tem nas mãos.
    """)

    st.subheader("Conteúdo")

    st.markdown("#### Módulo 1: Palestra: O Robô no Divã: Como Comportamento Afeta a IA")
    st.markdown("""Os alunos aprendem a associar comportamento à Ciência de Dados para aprimorar seus códigos e aumentar suas chances de promoção na empresa.
                Além disso, será criada a base para as discussões ao longo do curso.""", unsafe_allow_html=True)

    st.markdown("#### Módulo 2: Interface de usuário")
    st.markdown("""O programa de aulas será apresentado e a infra-estrutura necessária ao uso do Python será montada, bem como a interface com o usuário.
                 O código será feito com Visual Studio Code (ao invés de Jupyter) e os alunos ganham auto-confiança ao gerar rapidamente um aplicativo em Streamlit.""")

    st.markdown("#### Módulo 3: Deployment, Banco de Dados e Análise Exploratória")
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

    st.markdown("#### Módulo 8: Planejamento Estratégico e Finanças")  # cross-validation? comparacao de modelos? simulação de cenários?

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



with home3:
    st.subheader("Alunos: Motivação")
    st.markdown("""
    \nDecidimos fazer este projeto por ...
    """)

with home4:
    st.subheader("Alunos: Sumário Executivo")
    st.markdown("""
    \nO projeto ...
    """)

with home5:
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
