import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Importar datos en el contexto del país (Perú)
data = pd.read_csv('../Data/Data_Intercorp.csv')

# Mostrar datos de la compañía
print(data.head())

# Mostrar los datos visualmente (Gráficas)
data['Close'].plot()
plt.show()

# Mostrar datos de mañana para cada tiempo
data['Tomorrow'] = data['Close'].shift(-1)

# Crear un índice para caracteres si es más mañana
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

# Mostrar datos
print(data.head())

# Modelo RandomForest
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# División de datos
train = data.iloc[:-100]
test = data.iloc[-100:]

# Variables predictoras
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])

# Métricas
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Precisión de la prueba
print(precision_score(test['Target'], preds))

# Gráficas de predicción
combined = pd.concat([test['Target'], preds], axis=1)
combined.plot()
plt.show()

#----------------------------------------------------------------#

# Función de Retrospectiva
def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

def backtest(docs, model, predictors, start=57, step=30):
    all_predictions = []

    for i in range(start, docs.shape[0], step):
        train = docs.iloc[0:i].copy()
        test = docs.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Ver las predicciones
predictions = backtest(data, model, predictors)
print(predictions['Predictions'].value_counts())

# Nueva prueba de precisión
print(precision_score(predictions['Target'], predictions['Predictions']))

# Precisión para los valores bajos y altos
print(predictions['Target'].value_counts()/predictions.shape[0])

#----------------------------------------------------------------#

horizons = [2, 5, 60, 250]
new_predictors = []

for horizon in horizons:
    rolling_averages = data['Close'].rolling(window=horizon).mean()

    ratio_column = f'Close_Ratio_{horizon}'
    data[ratio_column] = data['Close'] / rolling_averages

    trend_column = f'Trend_{horizon}'
    data[trend_column] = data['Target'].shift(1).rolling(window=horizon).sum()

    new_predictors += [ratio_column, trend_column]

print(data.head())

#----------------------------------------------------------------#

# Nueva confianza del modelo
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['Target'])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name='Predictions')
    combined = pd.concat([test['Target'], preds], axis=1)
    return combined

predictions = backtest(data, model, new_predictors)
print(predictions['Predictions'].value_counts())
