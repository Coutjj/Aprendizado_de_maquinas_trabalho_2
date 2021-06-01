import pandas as pd
import math
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso


conjunto_treino_df = pd.read_csv(
    "eel891-202002-trabalho-2/conjunto_de_treinamento.csv"
)


variaveis_categoricas = [
    coluna for coluna in conjunto_treino_df.columns
    if conjunto_treino_df[coluna].dtype == 'object'
]

print("\nVariaveis categoricas\n")
print(variaveis_categoricas)

# ['tipo', 'bairro', 'tipo_vendedor', 'diferenciais']

print('\nVerificar a quantidade de amostras em diferenciais:\n')
print(conjunto_treino_df['diferenciais'].value_counts())

print('\nVerificar a quantidade de amostras em tipo_vendedor:\n')
print(conjunto_treino_df['tipo_vendedor'].value_counts())

print('\nVerificar a quantidade de amostras em bairro:\n')
print(conjunto_treino_df['bairro'].value_counts())

print('\nVerificar a quantidade de amostras em tipo:\n')
print(conjunto_treino_df['tipo'].value_counts())

print("\nVerificar cardinalidade para cada variavel categorica\n")

for var_categorica in variaveis_categoricas:
    print(
        var_categorica + ": "
        + str(len(conjunto_treino_df[var_categorica].unique()))
    )

print("\nDescartando variaveis desnecessarias\n")
remover = ['tipo', 'bairro', 'tipo_vendedor', 'diferenciais', 'Id']
conjunto_treino_df = conjunto_treino_df.drop(remover, axis=1)

colunas = conjunto_treino_df.columns

print('\nColunas:\n')
print(colunas)

print('\nCorrelacao Preco x Coluna:\n')
for coluna in colunas:
    print('%s = %6.3f' % (
        coluna,
        pearsonr(
            conjunto_treino_df[coluna],
            conjunto_treino_df['preco'])[0]
        )
    )
print()


dados_treino = conjunto_treino_df.iloc[:, :-1].to_numpy()
dados_alvo = conjunto_treino_df.iloc[:, -1].to_numpy()

print(dados_alvo)

x_treino, x_teste, y_treino, y_teste = train_test_split(
    dados_treino,
    dados_alvo,
    train_size=0.7,
    random_state=1543
)

print(len(dados_alvo))
escalador = StandardScaler()
# escalador = MinMaxScaler()
escalador.fit(x_treino)
x_treino = escalador.transform(x_treino)
x_teste = escalador.transform(x_teste)

print('  K  Resultado')
print(' --- ---------------')

for k in range(1, 9):
    # pf = PolynomialFeatures(degree=k)

    # pf = pf.fit(x_treino)
    # x_treino_poly = pf.transform(x_treino)
    # x_teste_poly = pf.transform(x_teste)

    # regressor_linear = LinearRegression()
    # regressor_linear = regressor_linear.fit(x_treino_poly, y_treino)

    # regressor_ridge = Ridge(alpha=1000)
    # regressor_ridge = regressor_ridge.fit(x_treino_poly, y_treino)

    regressor_sgd = SGDRegressor(
        loss='squared_loss',
        alpha=1.2,
        penalty='l2',
        # tol=1e-5,
        max_iter=100000
    )

    regressor_sgd = regressor_sgd.fit(x_treino, y_treino)

    y_resposta_teste = regressor_sgd.predict(x_teste)

    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)

    print(' %3d %15.4f' % (k, rmse_out))



# Resposta


conjunto_teste_df = pd.read_csv(
    "eel891-202002-trabalho-2/conjunto_de_teste.csv"
)

Ids_teste = conjunto_teste_df['Id']

conjunto_teste_df = conjunto_teste_df.drop(remover, axis=1)
dados_teste = conjunto_teste_df.iloc[:, :].to_numpy()

escalador = StandardScaler()
escalador.fit(dados_treino)
x_treino = escalador.transform(dados_treino)
x_teste = escalador.transform(dados_teste)


# pf = PolynomialFeatures(degree=1)

# pf = pf.fit(x_treino)
# x_treino_poly = pf.transform(x_treino)
# x_teste_poly = pf.transform(x_teste)

# regressor_linear = LinearRegression()
# regressor_linear = regressor_linear.fit(x_treino_poly, dados_alvo)

# regressor_ridge = Ridge(alpha=10000)
# regressor_ridge = regressor_ridge.fit(x_treino_poly, dados_alvo)

regressor_sgd = SGDRegressor(
        loss='squared_loss',
        alpha=1.2,
        penalty='l2',
        # tol=1e-5,
        max_iter=100000
    )

regressor_sgd = regressor_sgd.fit(x_treino, dados_alvo)

y_resposta_teste = regressor_sgd.predict(x_teste)

# compondo dataframes
resposta = pd.DataFrame(y_resposta_teste, columns=['preco'])
resposta = pd.concat([Ids_teste, resposta], axis=1, join='inner')

print(resposta)

resposta.to_csv('./resposta.csv', index=False)