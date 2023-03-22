# Formação Cientista de Dados - Novo Projeto com Feedback 2
# Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio
#
# Nome do Aluno: Victor Hugo Nishitani
#
#   O teste hidrostático extintor é um procedimento estabelecido pelas normas da ABNT NBR 
# 12962/2016, que determinam que todos os extintores devem ser testados a cada cinco anos, 
# com a finalidade de identificar eventuais vazamentos, além de também verificar a 
# resistência do material do extintor.
#   Com isso, o teste hidrostático extintor pode ser realizado em baixa e alta pressão, de
# acordo com estas normas em questão. O procedimento é realizado por profissionais técnicos 
# da área e com a utilização de aparelhos específicos e apropriados para o teste, visto que 
# eles devem fornecer resultados com exatidão.
#   Seria possível usar Machine Learning para prever o funcionamento de um extintor de 
# incêndio com base em simulações feitas em computador e assim incluir uma camada adicional 
# de segurança nas operações de uma empresa? Esse é o objetivo do Projeto com Feedback 2.
#   Usando dados reais disponíveis publicamente, seu trabalho é desenvolver um modelo de 
# Machine Learning capaz de prever a eficiência de extintores de incêndio.
# 
#   No link abaixo você encontra os dados necessários para o seu trabalho:
#            https://www.muratkoklu.com/datasets/vtdhnd07.php
#   
#   O conjunto de dados foi obtido como resultado dos testes de extinção de quatro chamas de 
# combustíveis diferentes com um sistema de extinção de ondas sonoras. O sistema de extinção 
# de incêndio por ondas sonoras consiste em 4 subwoofers com uma potência total de 
# 4.000 Watts. Existem dois amplificadores que permitem que o som chegue a esses subwoofers 
# como amplificado. A fonte de alimentação que alimenta o sistema e o circuito do filtro 
# garantindo que as frequências de som sejam transmitidas adequadamente para o sistema está 
# localizada dentro da unidade de controle. Enquanto o computador é usado como fonte de 
# frequência, o anemômetro foi usado para medir o fluxo de ar resultante das ondas sonoras 
# durante a fase de extinção da chama e um decibelímetro para medir a intensidade do som. 
# Um termômetro infravermelho foi utilizado para medir a temperatura da chama e da lata de 
# combustível, e uma câmera é instalada para detectar o tempo de extinção da chama. 
# Um total de 17.442 testes foram realizados com esta configuração experimental.
# Os experimentos foram planejados da seguinte forma:
#   - Três diferentes combustíveis líquidos e combustível GLP foram usados para criar a chama.
#   - 5 tamanhos diferentes de latas de combustível líquido foram usados para atingir 
#       diferentes tamanhos de chamas.
#   - O ajuste de meio e cheio de gás foi usado para combustível GLP.
#   Durante a realização de cada experimento, o recipiente de combustível, a 10 cm de 
# distância, foi movido para frente até 190 cm, aumentando a distância em 10 cm a cada vez. 
# Junto com o recipiente de combustível, o anemômetro e o decibelímetro foram movidos para 
# frente nas mesmas dimensões.
#   Experimentos de extinção de incêndio foram conduzidos com 54 ondas sonoras de frequências 
# diferentes em cada distância e tamanho de chama.
#   Ao longo dos experimentos de extinção de chama, os dados obtidos de cada dispositivo de 
# medição foram registrados e um conjunto de dados foi criado. 
# 
#   O conjunto de dados inclui as características do tamanho do recipiente de combustível 
# representando o tamanho da chama, tipo de combustível, frequência, decibéis, distância, 
# fluxo de ar e extinção da chama. Assim, 6 recursos de entrada e 1 recurso de saída serão 
# usados no modelo que você vai construir.
# 
#   A coluna de status (extinção de chama ou não extinção da chama) pode ser prevista usando 
# os seis recursos de entrada no conjunto de dados. Os recursos de status e combustível são 
# categóricos, enquanto outros recursos são numéricos.
#   
#   Seu trabalho é construir um modelo de Machine Learning capaz de prever, com base em novos 
# dados, se a chama será extinta ou não ao usar um extintor de incêndio.
#
# Dicionário de Dados:
#
# Dados de Propriedades e Descrições para Combustíveis Líquidos	e	LPG (GPL)
# 
# SIZE (Tamanho da Chama): 7 (cm=1), 12 (cm=2), 14 (cm=3), 16 (cm=4), 20 (cm=5),
#                          Ajuste de Meia Aceleração = 6 e Ajuste de Aceleração Total = 7
# FUEL (Tipo de Combustível): 1-Gasolina, 2-Querosene, 3-LPG/GPL e 4-Diluente
# DISTANCE (Distância):	 10-190 (cm)
# DESIBEL (Decibéis):	 72-113	(dB)
# AIRFLOW (Fluxo de Ar):	 0-17 (m/s)
# FREQUENCY (Frequência): 1-75 (Hz)
# STATUS (Extinção da Chama): 0 - Indica o estado de não extinção da chama
#                             1 - Indica o estado de extinção da chama


# Configurando o diretório de trabalho
setwd("/Users/nishi/Desktop/FCD/BigDataRAzure/Cap20/Projeto02")
getwd()

# Carrega os pacotes na sessão R
library(Hmisc)
library(psych)
library(ggplot2)
library(caTools)
library(class)
library(gmodels)

## Etapa 1 - Coletando os Dados
##### Carga dos Dados ##### 

# Carregamos o dataset antes da transformação
dados <- read.csv("Acoustic_Extinguisher_Fire_Dataset.csv", stringsAsFactors = F, 
                  sep = ";", dec = ",", header = T)


## Etapa 2 - Pré-Processamento
##### Análise Exploratória dos Dados - Limpeza e Organização de Dados ##### 

# Visualizamos os dados
View(dados)
dim(dados)
str(dados)

## Exploramos as variáveis categóricas

# Verificamos os valores únicos na variável categórica "STATUS"
length(unique(dados$STATUS))
hist(dados$STATUS)
prop.table(table(dados$STATUS))

# Os dados da variável "STATUS" parecem balanceados com valores na mesma proporção
# 0 - 0.5021786 e 1 - 0.4978214. Neste caso, não precisaremos fazer o balanceamento
# das classes

# Verificamos os valores únicos na variável categórica "FUEL"
length(unique(dados$FUEL))
table(dados$FUEL)
x <- as.factor(dados$FUEL)
data.frame(levels = unique(x), value = as.numeric(unique(x)))

# Como a variável "FUEL" é categórica, vamos transformá-la para sua representação
# numérica usando label enconding. Logo, vamos criar um novo dataframe para
# não modificar o dataset original
dados1 <- dados
dados1$FUEL <- as.integer(factor(dados$FUEL))

# Verificamos se temos valores ausentes
sum(is.na(dados1))
sum(!complete.cases(dados1))
prop.table(table(is.na(dados1)))

# Função que exibe as colunas com valores ausentes no dataframe
nacols <- function(df) {
  colnames(df)[unlist(lapply(df, function(x) any(is.na(x))))]
}
nacols(dados1)

# Verificamos se temos valores vazios
colSums(is.na(dados1) | dados1 == "")

# Verificamos se temos valores duplicados
sum(duplicated(dados1))


## Exploramos as variáveis numéricas
summary(dados1)
str(dados1)
apply(dados1[-7],2,sd)

# Construímos um Boxplot apenas para confirmar que não há possíveis outliers 
# na variável AIRFLOW
boxplot(dados1$AIRFLOW, main = "Boxplot do Fluxo de Ar", 
        ylab = "Fluxo de Ar")

# Construímos um Boxplot apenas para confirmar que não há possíveis outliers 
# na variável FREQUENCY
boxplot(dados1$FREQUENCY, main = "Boxplot da Frequência", 
        ylab = "Frequência")

# Criamos um Histograma para analisar a distribuição dos dados das variáveis
hist.data.frame(dados1) # Pacote Hmisc

# Fazemos a correlação entre as variáveis
cor(dados1)

# Este gráfico fornece mais informações sobre o relacionamento entre as variáveis
pairs.panels(dados1) # Pacote psych


## Etapa 3: Treinando o modelo e Criando o Modelo Preditivo no R

# Vamos dividir os dados em treino e teste, sendo 70% para dados de treino e 
# 30% para dados de teste
set.seed(123)
split = sample.split(dados1$STATUS, SplitRatio = 0.70)
dados_treino = subset(dados1, split == TRUE)
dados_teste = subset(dados1, split == FALSE)

# Verificando o número de linhas
nrow(dados_treino)
nrow(dados_teste)

# Visualizando as 5 primeiras posições dos dados de treino e teste
head(dados_treino, 5)
head(dados_teste, 5)

### Feature Scaling - Padronização dos Dados ###

# Criando novos datasets de treino e de teste

# Usando a função scale() para padronizar o z-score 
dados_treino1 <- as.data.frame(scale(dados_treino[-7]))

# Confirmando transformação realizada com sucesso
dados_teste1 <- as.data.frame(scale(dados_teste[-7]))

dim(dados_treino)
dim(dados_teste)

dados_treino_labels <- dados_treino[1:12209, 7] 
dados_teste_labels <- dados_teste[1:5233, 7]
length(dados_treino_labels)
length(dados_teste_labels)

# Criando o modelo
modelo_knn_v1 <- knn(train = dados_treino1, 
                     test = dados_teste1,
                     cl = dados_treino_labels, 
                     k = 21)

# A função knn() retorna um objeto do tipo fator com as previsões para cada 
# exemplo no dataset de teste
summary(modelo_knn_v1)


## Etapa 4: Avaliando e Interpretando o Modelo

# Criando uma tabela cruzada dos dados previstos x dados atuais
# Usaremos amostra com 5233 observações: length(dados_teste_labels)
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE)

# Interpretando os Resultados
# A tabela cruzada mostra 4 possíveis valores, que representam os falso/verdadeiro positivo e negativo
# Temos duas colunas listando os labels originais nos dados observados
# Temos duas linhas listando os labels dos dados de teste

# Temos:
# Cenário 1: Não Extinção da Chama (Observado) x Não Extinção da Chama (Previsto) - 
#            2498 casos (95%) - true negative (o modelo acertou)
# Cenário 2: Não Extinção da Chama (Observado) x Extinção da Chama (Previsto) - 
#            130 casos (5%) - false positive (o modelo errou)
# Cenário 3: Extinção da Chama (Observado) x Não Extinção da Chama (Previsto) - 
#            138 casos (5%) - false negative (o modelo errou)
# Cenário 4: Extinção da Chama (Observado) x Extinção da Chama (Previsto) - 
#            2467 casos (95%) - true positive (o modelo acertou)

# Lendo a Confusion Matrix (Perspectiva de ter ou não a doença):

# True Negative  = nosso modelo previu que NÃO houve Extinção da Chama e os dados mostraram 
#                  que NÃO, houve Extinção da Chama.
# False Positive = nosso modelo previu que houve Extinção da Chama e os dados mostraram 
#                  que NÃO, houve Extinção da Chama.
# False Negative = nosso modelo previu que NÃO houve Extinção da Chama e os dados mostraram 
#                  que SIM, houve Extinção da Chama.
# True Positive  = nosso modelo previu que houve Extinção da Chama e os dados mostraram 
#                  que SIM, houve Extinção da Chama.

# Falso Positivo - Erro Tipo I
# Falso Negativo - Erro Tipo II

# Taxa de acerto do Modelo: 95% (acertou 4965 em 5233 e errou apenas 268) o que é ótimo
# para uma primeiro modelo.


## Etapa 5: Otimizando a Performance do Modelo




