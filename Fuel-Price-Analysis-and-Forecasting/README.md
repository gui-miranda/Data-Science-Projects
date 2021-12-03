# PREVISOR DO PREÇO DE COMBUSTIVEIS - ANÁLISE DE SÉRIES TEMPORAIS UTILIZANDO REDES NEURAIS RECORRENTES
Este é um projeto cujo objetivo é analisar e desenvolver um modelo previsor para os valores de venda de combustiveis. São utilizadas para isso métodos avançados em machine learning, de forma que o modelo final obtido se baseia numa Rede Neural Recorrente (LSTM)
****


## 🔍 MOTIVAÇÃO

Com a recente alta dos combustíveis nos postos brasileiros,empresas que tem sua receita diretamente ligada ao uso de automóveis podem enfrentar dificuldades do ponto de vista do gerenciamento de custos de suas operações. Nesse contexto, a possibilidade de estimar com precisão o valor dos combustíveis num futuro comercialmente próximo mostra-se como uma forma eficaz de evitar prejuízos decorrentes de uma ineficiência em se computar corretamente os custos envolvidos.
****

## :books: SOBRE O PROJETO

O objetivo do presente projeto é a construção de um Modelo de Machine Learning que seja capaz de, com base em padrões visualizados nos dados históricos,produzir uma previsão confiável para o valor de revenda dos combustiveis mais comumente utilizados.

Esse projeto é baseado nos dados históricos fornecidos pela  Agência Nacional do Petróleo, Gás Natural e Biocombustíveis, que são disponibilizados pelo próprio governo federal atráves do site : (https://dados.gov.br/). Foram utilizados três diferentes conjuntos de dados para a construção das análises : Série histórica de preços de combustíveis , Vendas de derivados de petróleo e biocombustíveis,Produção de petróleo e gás natural nacional.

A principio os 3 conjuntos apresentavam dados relativos á todo o território Brasileiro, mas para que o projeto pudesse estar mais alinhado com a real situação economica de cada estado, os dados utilizados resultantes após o processo de limpeza são referentes à produção/venda/valor médio de revenda para o estado de São Paulo. 
 
Para a criação de um modelo de previsão o método utilizado foi construção de uma Rede Neural Recorrente (ou LSTM) utilizando o framework Pytorch. Além disso, como um parâmetro de comparação , uma análise da utilização das médias móveis para a previsão dos valores também foi realizada.
****
## :rotating_light: RESUMO DOS RESULTADOS

Em linhas gerais, pode-se concluir que a realização das previsões utilizando as redes neurais do tipo LSTM foi um sucesso. As acurácias na determinação do preço médio de cada combustivel no mês seguinte foram as seguintes:
 * Para a Gasolina -> 97.13%
 
 * Para o Etanol    -> 95.23%
***
## 📌 CONTEÚDO
 * 01 Data Cleaning
    * Carregamentos dos dados
    
    * Limpeza e tratamento dos dados presentes em cada conjunto
    
    * União dos três conjunto de dados em um só dataset
                  
                  
 * 02 Exploratory Data Analysis
    * Levantamento dos valores históricos para cada feature em análise
    
    * Levantamento da distribuição dos dados análisados 
                  
                  
 * 03 Feature Engineering
     * Determinação das correlações entre os parâmetros utilizados e a variável algo (Valor de Venda)
     
     * Decomposição da Série Temporal dos Valores de Venda
     
     * Criação de novas features relacionados às datas
     
     * Aplicação de uma Tranformação de Potência 
     
     * Scalling dos dados utilizados
                  
                  
 * 04 Model Building
    * Construção de Médias Móveis para gerar previsões 
     
    * Construção e Validação da LSTM
    
    * Análise comparativa entre os métodos construídos 
 
 * 05 Conclusão
    * Análise das previsões geradas do as LSTM.
    
****

## DATA CLEANING
Após o processo de filtragem dos dados para o estado de São Paulo, o conjunto de dados possuia as seguintes features e dimensões:

![DF](https://github.com/gui-miranda/Projecao-do-Preco-de-Combustiveis/blob/main/Imagens/dataset.JPG)

**[201,6]**

 Onde as váriaveis em questão são:
   * valores_g : Valor de revenda p/ gasolina em [R$]
   * valores_e : Valor de revenda p/ etanol em [R$]
   * vendas_g  : Total de vendas p/ gasolina em [m³]
   * vendas_e  : Total de vendas p/ etanol em [m³]
   * prod_pe   : Produção de petróleo em [m³]
   * vendas_e  : Produção de etanol em [m³]
   
Deve-se destacar que , por se tratarem de dados fornecidos pelo governo federal, esses se encontravam bem estruturados e na ausência de valores nulos. Pode-se visualizar esse fato atráves de um heatmap:

![null_value](https://github.com/gui-miranda/Projecao-do-Preco-de-Combustiveis/blob/main/Imagens/null_values.JPG)

****
## EXPLORATORY DATA ANALYSIS
Essa seção não se tratará de uma minunciosa análise dos dados trabalhados com o intuito de gerar possíveis insights. Ao invés disso uma abordagem mais objetiva será adotada, de forma que a ideia principal é entender melhor a distribuição e o compartamento desses dados ao longo tempo. Essa análise é de suma importância pois fornecerá informações preciosas sobre a presença de outliers, ruídos e irregularidades que poderão levar a etapas adicionais na preparação dos dados antes do treinamento dos modelos.

Dessa forma, os resultados obtidos para o comportamento temporal das séries de valores e suas respectivas distruibuições são:

![image](https://user-images.githubusercontent.com/82520450/141472718-1967a797-4e55-479e-bd7e-86e69dbb6055.png)

Além disso, utilizando o método "Seasonal and Trend decomposition using Loess" (STL) para decompor a série temporal dos valores de venda , temos os seguintes resultados:
* Para a Gasolina
![image](https://user-images.githubusercontent.com/82520450/144428905-2804d5e6-916c-441c-8c01-fc6137d890c6.png)

* Para o Etanol
![image](https://user-images.githubusercontent.com/82520450/144428981-43911c3b-5149-4bdb-a9ec-d4da3f499d99.png)


Quanto à essa análise é importante destacar 3 pontos:
 * A presença de linhas de tendência não lineares e variantes no tempo. Tendências não lineares podem acarretar em problemas nos tratamentos futuros, de maneira que análises envolvendo a diferenciação de 1ª ordem da série ficam comprometidas.
 
 * É possível visualizar efeitos sazonais significantes ao longo dos períodos. Tão importante quanto isso é o fato de que a sazonalinade sofreu variações na sua forma e amplitude ao longo do período observado.
 
 * Os residuos nos últimos períodos sofreram um aumento, oque pode significar que o simultâneo aumento da tendência ocasionou desvios consideráveis em relação ao ciclo sazonal.

Utilizar o STL mostrou-se mais eficiente do que os métodos convecionais de decomposição, isso porque é possivel computar as variações que a sazonalidade sofre ao lonogo do tempo. Além disso, outra característica desse método é sua capacidade em lidar com a presença de outliers,de forma que essas variações excêntricas não afetam as componentes da decomposição.
****
## FEATURE ENGINEERING
A partir das informações levantandas durante a etapa de exploração do dados, pode-se então ter uma ideia de quais processos podem beneficiar o aprendizado do modelo:
  
   **1**. Seleção das features a serem utilizadas com base nas sua correlações 
  
   **2**. Criação de features relacionadas à data (Devido aos efeitos de Sazonalidade)
   
   **3**. Aplicação de uma transformação de potência p/ redução dos ruídos e Normalização dos dados 

   * ## 1. SELEÇÃO DAS FEATURES
     
     Como critério para se decidir quais features possuiam um bom desempenho em descrever os valores alvos ('Valores de Venda'), utilizou-se o método "Pairwise Correlation" que
     afere as correlações lineares entre cada uma das features presentes no conjunto. 
     
     A matrix de correlação obtida é a seguinte:
     
     ![image](https://user-images.githubusercontent.com/82520450/141482458-1da54989-b871-426e-978b-6165d97e9864.png)
    
     É importante também analisar as correlações entre as variáveis em uma perspectiva geral e não só linear. Para tal fim, uma análise visual de como as variáveis se relacionam pode ajudar numa melhor compreensão do conjunto de dados.
     
     ![image](https://user-images.githubusercontent.com/82520450/141483362-4b8dab86-0ff2-4aaf-96c9-0e7ad5f4683e.png)

      Portanto, é possível concluir que as features possuem uma boa capacidade de descrever os nossos valores alvo e dessa forma não há necessidade de fazer o descarte de nenhuma delas nessa etapa. 
      
   * ## 2. CRIANDO VARIÁVEIS DE DATA
   O objetivo dessa etapa é utilizar as informações de data presentes no dataset para computar os efeitos sazonais encontrados na séries. Entretanto, diferentemente das features anteriores, a data é uma váriavel cíclica oque acrescenta uma maior complexidade no seu processamento para alimentação do modelo.
   
   Para realizar esse processamento foram utilizadas transformações trigonomètricas. Isso porque as funções trigonométricas são cíclicas, e isso permite que o modelo seja capaz de identificar "proximidades" entre duas datas sequenciais. Dessa maneira, as informações de data (como o mês) são normalizadas e utilizadas como o parâmetro "angular" de uma função trigonomètrica, e o resultado dessas operações é utilizado como uma nova feature para o modelo.
   
   A seguir seguem as correlações obtidas para as variáveis criadas:
   
   ![image](https://user-images.githubusercontent.com/82520450/141487874-c0bc9832-b04d-4116-b45d-0e0e2135563a.png)
   

* ## 3. TRANSFORMAÇÃO DE POTÊNCIA E NORMALIZAÇÃO
  Com o objetivo de diminuir o ruído (ou resíduos), presente nos dados analisados , aplicou-se uma transformação de potência de modo que tal processo possibilita um melhor aprendizado dos modelos e ainda assim conserva as suas generalidades.
  A transformação escolhida foi a logarítimica, de modo que o valor resultante para cada uma das features (exceto as relacionadas á data) foram dadas por: 
  
   ![image](https://user-images.githubusercontent.com/82520450/141505688-d40a45f0-9a26-4d31-b106-8280abf3d89d.png)

  Por fim, a normalização do conjunto de dados se fez necessária. Isso porque valores em diferentes ordens de grandeza estão operando simultaneamente, dessa forma é necessário colocá-los todos na mesma escala de grandeza. Para realizar essa operação utilou-se a convencional função MinMaxScaler(), presente na biblioteca Sklearn. 
****  
  ## MODEL BUILDING
  
   ### MÉDIAS MÓVEIS
  Inicialmente, uma análise utilizando as médias móveis como modelos previsores para o valor de venda dos combustiveis foi construida. É importante que essa análise seja realizado antes da construção das redes neurais pois ela permite um entendimento maior sobre como o quão dificil é descrever o comportamento da série. Além disso, as média móveis permitem que seja estabelicido um parâmetro inicial para optmização da rede neural construída.
  
  Foram traçadas 3 diferentes séries, cada uma correspondendo à média dos últimos 2,3 e 4 meses respectivamente. Os resultados para previsões geradas nos últimos 12 meses foi o seguinte:
  
  ![image](https://user-images.githubusercontent.com/82520450/144492132-b7d11094-6620-4c67-ac3d-873cd8ad65a1.png)
  
  Dessa forma pode-se concluir que a série dos valores do etanol apresenta maiores oscilações no ultimo ano, e portanto haverá um maior esforço em descrevê-la adequadramente.
  
 ### LSTM
 Como já discutido anteriomente, a classe de rede neural utilizada será a Long Short-Term Memory ou LSTM.Apesar do seu propósito inicial de ser uma ferramenta para o processamento de textos, esse tipo de rede tem ganhado grande destaque devido à sua eficácia em realizar previsões de séries temporais. Isso acontece pois, uma LSTM têm mecanismos internos que podem aprender quais dados em uma sequência são importantes para manter ou descartar, e então podem passar as informações relevantes pela longa cadeia de sequências para fazer previsões. É possível compreender melhor o funcionamento de uma LSTM através do diagrama a seguir:
 
![image](https://user-images.githubusercontent.com/82520450/144466735-a8c3a158-d576-4293-b24d-6f8e910af518.png)

 * ## MODELOS PREVENDO 1 MÊS À FRENTE:
 Os primeiros dois modelos foram construídos com o intuito de, a partir de todos os dados registrados no último mês, gerar uma previsão para o valor médio de venda dos combustíveis no mês seguinte.
 
 Uma análise detalhada em busca da melhor configuração de rede para representar as séries foi necessária, de modo que modelos individuais foram construídos para a gasolina e etanol. Além disso ambos foram válidados utilizando os valores registrados ao longo dos últimos 12 meses, e as métricas utilizadas foram o MAPE (Erro Percentual Médio Absoluto) e o MAE (Erro Médio Absoluto).
 
 Os resultados obtidos para cada um dos modelos foram os seguintes:
 
![image](https://user-images.githubusercontent.com/82520450/144489611-d61bd951-3f87-42d9-aee8-18be08a65d38.png)

* ## COMPARANDO OS ERROS
A partir das médias móveis e dos modelos construidos, pode-se então realizar uma comparação entre suas acurácias e definir portanto se houve uma diferença significante nos resultados.
  
  Têm-se então :

 ![image](https://user-images.githubusercontent.com/82520450/144482822-628e02b2-3204-4527-9b9d-2b3cb9aac06f.png)

 ![image](https://user-images.githubusercontent.com/82520450/144482557-d0975672-39da-4e9a-9c8c-b22712bc7896.png)
 
 Fica evidente portanto que os erros gerados pelas LSTM são significantemente menores do que os das Médias Móveis
 
 * ## CONCLUSÃO
 Por fim, é possível analisar as previsões realizadas por cada um dos modelos :
 
 Para a Gasolina, os seguinte resultados foram obtidos:
 ![image](https://user-images.githubusercontent.com/82520450/144468464-b067ae51-be24-4ec3-bca1-ad10f4752733.png)
 
 Já para o Etanol, os valores resultantes foram:
 ![image](https://user-images.githubusercontent.com/82520450/144468908-5542cd8f-c881-4372-a29f-15e464a9c806.png)
 
 
 
 


