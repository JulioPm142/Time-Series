import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

# Configurando o estilo do matplotlib
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.switch_backend('Agg')  # Para ambientes sem GUI

# Carregando dados
df = pd.read_csv('MLTempDataset.csv')

# Renomeando e convertendo a coluna de data
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.rename(columns={'Datetime': 'date'})
df = df.set_index('date')

# Calculando a média diária de temperatura
daily_df = df.resample('D').mean()

# Renomeando a coluna de temperatura para 'temperatura' para consistência com o exemplo
daily_df = daily_df.rename(columns={'DAYTON_MW': 'temperatura'})

# Plotando os dados
fig, ax = plt.subplots(figsize=(9, 4))
daily_df['temperatura'].plot(ax=ax, label='temperatura')
ax.legend()
plt.savefig('daily_temp_plot.png')

# Verificando valores nulos
print(f'Number of rows with missing values: {daily_df.isnull().any(axis=1).mean()}')
print((daily_df.index == pd.date_range(start=daily_df.index.min(), end=daily_df.index.max(), freq=daily_df.index.freq)).all())

# Dividindo dados em treino e teste
steps = 36
data_train = daily_df[:-steps]
data_test = daily_df[-steps:]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

# Plotando dados de treino e teste
fig, ax = plt.subplots(figsize=(9, 4))
data_train['temperatura'].plot(ax=ax, label='train')
data_test['temperatura'].plot(ax=ax, label='test')
ax.legend()
plt.savefig('train_test_split.png')

# Configurando e treinando o modelo
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(max_depth=10, n_estimators=50, random_state=123),
                lags = 12
             )
forecaster.fit(y=data_train['temperatura'])
forecaster

# Fazendo predições
predictions = forecaster.predict(steps=steps)
predictions.head(5)

# Plotando predições
fig, ax = plt.subplots(figsize=(9, 4))
data_train['temperatura'].plot(ax=ax, label='train')
data_test['temperatura'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions', color='green')
ax.legend()
plt.savefig('predictions_plot.png')
