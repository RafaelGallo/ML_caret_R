# Usando o Pacote Caret Para Criar Modelos de Machine Learning em R
# http://topepo.github.io/caret/index.html

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
setwd("C:/FCD/BigDataRAzure/Cap12")
getwd()

# Instalando os pacotes
install.packages("caret")
install.packages("randomForest")

# Carregando os pacotes
library(caret)
library(randomForest)
library(datasets)

# Usando o dataset mtcars
head(mtcars)

# Criando a função caret para devisão dados
model_train <- createDataPartition(y = mtcars$mpg,
                                  p = 0.7,
                                  list = FALSE)
model_train

# Criando treinamento, teste para modelo ML
train <- mtcars[model_train,]
test <- mtcars[-model_train,]

dim(train)
dim(test)

# Treinando modelo
names(getModelInfo())

# A importância das variáveis para criação modelo
modelo_N1 <- train(mpg ~., data = train, method = "lm")
modelo_N1
summary(modelo_N1)

# Modelo regressão linear
modelo_regressao_linear <- train(mpg ~ wt + 
                                   hp + 
                                   qsec + 
                                   drat,
                                 data = train,
                                 method = "lm")
modelo_regressao_linear
summary(modelo_regressao_linear)

# Modelo Random forest
modelo_random_forest <- train(mpg ~ wt + 
                              hp + 
                              qsec + 
                              drat, 
                              data = train, 
                              method = "rf")
modelo_random_forest
summary(modelo_random_forest)

# Resumo dos modelos ML
summary(modelo_regressao_linear)
summary(modelo_random_forest)

# Modelo ajustado
control_modelo <- trainControl(method = "cv", number = 10)
control_modelo
modelo_v1 <- train(mpg ~wt + hp + qsec + drat,
                   data = train,
                   method = "lm",
                   trControl = control_modelo,
                   metric = "Rsquared")
modelo_v1
summary(modelo_v1)

# Residuos
resd <- resid(modelo_v1)
resd

# Previsoes do modelo
model_pred_1 <- predict(modelo_regressao_linear, test)
model_pred_2 <- predict(modelo_random_forest, test)

# Gráfico
plot(test$mpg, model_pred_1)
plot(test$mpg, model_pred_2)

# Plot das variáveis mais relevantes no modelo
plot(varImp(modelo_N1))
