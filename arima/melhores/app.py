import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# Usando o backend Agg
plt.switch_backend('Agg')

# Lendo o CSV
df = pd.read_csv('MLTempDataset.csv')

# Convertendo a coluna 'Datetime' para o formato datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Definindo a coluna 'Datetime' como índice
df.set_index('Datetime', inplace=True)

# Calculando a média diária de temperatura
daily_df = df.resample('D').mean()

# Plotando a série temporal
plt.figure(figsize=(10, 6))
plt.plot(daily_df.index, daily_df['DAYTON_MW'], label='Temperatura Média Diária')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.title('Temperatura Média Diária')
plt.legend()
plt.savefig('temperatura_media_diaria.png')

train_size = int(len(daily_df) * 0.80)
train, test = daily_df.iloc[:train_size], daily_df.iloc[train_size:]

# Manualmente configurando p, d, q
p = 45  # substitua com seu valor escolhido
d = 1  # substitua com seu valor escolhido
q = 1  # substitua com seu valor escolhido


# melhor:
# p = 50
# d = 1  
# q = 1 

# Treinando o modelo ARIMA
model = ARIMA(train['DAYTON_MW'], order=(p, d, q))
model_fit = model.fit()

# Fazendo previsões
forecast = model_fit.forecast(steps=len(test))

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['DAYTON_MW'], label='Treino')
plt.plot(test.index, test['DAYTON_MW'], label='Teste')
plt.plot(test.index, forecast, label='Previsão', color='red')
plt.xlabel('Data')
plt.ylabel('Temperatura')
plt.title('Previsão de Temperatura Média Diária com ARIMA')
plt.legend()
file_name = f'previsao_arima_P{p}_D{d}_Q{q}.png'
plt.savefig(file_name)


# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# fig, axes = plt.subplots(1, 2, figsize=(16, 4))
# plot_acf(daily_df['DAYTON_MW'].dropna(), lags=40, ax=axes[0])
# plot_pacf(daily_df['DAYTON_MW'].dropna(), lags=40, ax=axes[1])
# plt.savefig('acf_pacf_analysis.png')

