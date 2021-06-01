import pandas as pd
import math
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor


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


########### Dividindo em regioes politico administrativas de recife ######

dict_regioes = {
    'RPA1': [
        'Recife',
        'Centro',
        'Boa Vista',
        'Cabanga',
        'Coelhos',
        'Ilha do Leite',
        'Ilha Joana Bezerra',
        'Paissandu',
        'Sto Amaro',
        'Sto Antonio',
        'Antonio',
        'S Jose',
        'Soledade',
        'Piedade'
    ],
    'RPA2': [
        'Agua Fria',
        'Alto Santa Terezinha',
        'Arruda',
        'Beberibe',
        'Bomba do Hemeterio',
        'Cajueiro',
        'Campina do Barreto',
        'Campo Grande',
        'Encruzilhada',
        'Fundao',
        'Hipodromo',
        'Linha do Tiro',
        'Peixinhos',
        'Ponto de Parada',
        'Porto da Madeira',
        'Rosarinho',
        'Torreao'
    ],
    'RPA3': [
        'Aflitos',
        'Alto do Mandu',
        'Alto Jose Bonifacio',
        'Alto José do Pinho',
        'Apipucos',
        'Brejo da Guabiraba',
        'Brejo de Beberibe',
        'Casa Amarela',
        'Casa Forte',
        'Corrego do Jenipapo',
        'Derby',
        'Dois Irmaos',
        'Espinheiro',
        'Gracas',
        'Guabiraba',
        'Jaqueira',
        'Macaxeira',
        'Mangabeira',
        'Monteiro',
        'Morro da Conceição',
        'Nova Descoberta',
        'Parnamirim',
        'Passarinho',
        'Pau-Ferro',
        'Poco',
        'Poco da Panela',
        'Santana',
        'Sítio dos Pintos',
        'Tamarineira',
        'Vasco da Gama'
    ],
    'RPA4': [
        'Caxanga',
        'Cid Universitaria',
        'Cordeiro',
        'Engenho do Meio',
        'Ilha do Retiro',
        'Iputinga',
        'Madalena',
        'Beira Rio',
        'Torre',
        'Prado',
        'Torrões',
        'Varzea',
        'Zumbi'
    ],
    'RPA5': [
        'Afogados',
        'Areias',
        'Barro',
        'Bongi',
        'Caçote',
        'Coqueiral',
        'Curado',
        'Estância',
        'Jd S Paulo',
        'Jiquia',
        'Mangueira',
        'Mustardinha',
        'San Martin',
        'Sancho',
        'Tejipio',
        'Toto'
    ],
    'RPA6': [
        'Boa Viagem',
        'Setubal',
        'Brasilia Teimosa',
        'Cohab',
        'Ibura',
        'Imbiribeira',
        'Lagoa do Araca',
        'Ipsep',
        'Jordão',
        'Pina'
    ]

}

dict_estados_regioes = dict()
for rpa in dict_regioes:
    for estado in dict_regioes[rpa]:
        dict_estados_regioes[estado] = rpa

conjunto_treino_df['RPA'] = (
    conjunto_treino_df['bairro'].map(dict_estados_regioes)
)

print('\nVerificar a quantidade de amostras em RPA:\n')
print(conjunto_treino_df['RPA'].value_counts())

conjunto_treino_df = pd.get_dummies(
    conjunto_treino_df,
    columns=['RPA'],
    prefix='RPA',
)
###################################################################

####### Coeficiente de pearson em todas as colunas ################
remover = ['Id', 'tipo', 'bairro', 'tipo_vendedor', 'diferenciais']
conjunto_treino_calculo_pearson = conjunto_treino_df.drop(remover, axis=1)

pearson_dict = dict()
colunas = conjunto_treino_calculo_pearson.columns

print('\nCorrelacao Preco x Coluna:\n')
for coluna in colunas:
    pearson_dict[coluna] = pearsonr(
        conjunto_treino_calculo_pearson[coluna],
        conjunto_treino_calculo_pearson['preco']
    )[0]

pearson_dict_sorted = sorted(
    pearson_dict.items(),
    key=lambda x: abs(x[1]),
    reverse=True
)

pearson_dict_sorted = dict(pearson_dict_sorted)

for item in pearson_dict_sorted:
    print('{:<15} = {:.4f}'.format(item, pearson_dict[item]))

print()

#######################################################

colunas = conjunto_treino_df.columns
print('\nColunas:\n')
print(colunas)

print("\nDescartando variaveis desnecessarias\n")
remover = [
            'Id',
            'tipo',
            'bairro',
            'tipo_vendedor',
            # 'quartos',
            # 'suites',
            # 'vagas',
            # 'area_util',
            'area_extra',
            'diferenciais',
            'churrasqueira',
            'estacionamento',
            'piscina',
            'playground',
            'quadra',
            's_festas',
            's_jogos',
            's_ginastica',
            'sauna',
            'vista_mar',
            # 'preco',
            'RPA_RPA1',
            'RPA_RPA2',
            'RPA_RPA3',
            'RPA_RPA4',
            'RPA_RPA5',
            'RPA_RPA6'
]

conjunto_treino_df = conjunto_treino_df.drop(remover, axis=1)
print(conjunto_treino_df)

dados_treino = conjunto_treino_df.iloc[:, :-1].to_numpy()
dados_alvo = conjunto_treino_df.iloc[:, -1].to_numpy()

x_treino, x_teste, y_treino, y_teste = train_test_split(
    dados_treino,
    dados_alvo,
    train_size=0.7,
    random_state=1543
)

escalador = StandardScaler()
escalador.fit(x_treino)
x_treino = escalador.transform(x_treino)
x_teste = escalador.transform(x_teste)

print('K  Resultado')
print('--- ---------------')

for k in range(1, 5):

    pf = PolynomialFeatures(degree=k)
    pf = pf.fit(x_treino)
    x_treino_poly = pf.transform(x_treino)
    x_teste_poly = pf.transform(x_teste)

    # regressor_linear = LinearRegression()
    # regressor_linear = regressor_linear.fit(x_treino_poly, y_treino)

    # regressor_ridge = Ridge(alpha=500, solver='lsqr')
    # regressor_ridge = regressor_ridge.fit(x_treino_poly, y_treino)
    # regressor_ridge = regressor_ridge.fit(x_treino_poly, y_treino)
    # y_resposta_teste = regressor_ridge.predict(x_teste_poly)

    # RF_regressor = RandomForestRegressor(n_estimators=k)
    # RF_regressor.fit(x_treino, y_treino)
    # y_resposta_teste = RF_regressor.predict(x_teste)

    regressor_sgd = SGDRegressor(
        loss='squared_loss',
        alpha=1.2,
        penalty='l2',
        # tol=1e-5,
        max_iter=100000,
        random_state=0
    )
    regressor_sgd.fit(x_treino, y_treino)
    y_resposta_teste = regressor_sgd.predict(x_teste)

    mse_out = mean_squared_error(y_teste, y_resposta_teste)
    rmse_out = math.sqrt(mse_out)
    r2_out = r2_score(y_teste, y_resposta_teste)
    rmspe = np.sqrt(np.mean(np.square(((y_teste - y_resposta_teste) / y_teste)), axis=0))

    print(' %3d %15.4f' % (k, rmspe))

print()
# Resposta


conjunto_teste_df = pd.read_csv(
    "eel891-202002-trabalho-2/conjunto_de_teste.csv"
)

Ids_teste = conjunto_teste_df['Id']

#Dividindo em regioes politico administrativas de recife
conjunto_teste_df['RPA'] = conjunto_teste_df['bairro'].map(dict_estados_regioes)

# print('\nVerificar a quantidade de amostras em bairro:\n')
# print(conjunto_teste_df['RPA'].value_counts())

# print('\nOne-hot encoding regiao\n')

conjunto_teste_df = pd.get_dummies(
    conjunto_teste_df,
    columns=['RPA'],
    prefix='RPA'
)

conjunto_teste_df = conjunto_teste_df.drop(remover, axis=1)

dados_teste = conjunto_teste_df.iloc[:, :].to_numpy()

escalador = StandardScaler()
escalador.fit(dados_treino)
x_treino = escalador.transform(dados_treino)
x_teste = escalador.transform(dados_teste)


pf = PolynomialFeatures(degree=2)

pf = pf.fit(x_treino)
x_treino_poly = pf.transform(x_treino)
x_teste_poly = pf.transform(x_teste)

# regressor_linear = LinearRegression()
# regressor_linear = regressor_linear.fit(x_treino_poly, dados_alvo)

regressor_ridge = Ridge(alpha=3200, solver='lsqr')
regressor_ridge = regressor_ridge.fit(x_treino_poly, dados_alvo)

# regressor_sgd = SGDRegressor(
#         loss='squared_loss',
#         alpha=1.2,
#         penalty='l2',
#         # tol=1e-5,
#         max_iter=100000
#     )

y_resposta_teste = regressor_ridge.predict(x_teste_poly)

# compondo dataframes
resposta = pd.DataFrame(y_resposta_teste, columns=['preco'])
resposta = pd.concat([Ids_teste, resposta], axis=1, join='inner')

print(resposta)

resposta.to_csv('./resposta.csv', index=False)
