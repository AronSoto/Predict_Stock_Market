import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

#Importing data in the country context(Perú)
data = pd.read_csv('../Data/Data_Intercorp.csv')

#Showing data a company
data

#Show the data visually (Gráficas)
data['Close'].plot()

#Show the data tomorrow for every time
data['Tomorrow'] = data['Close'].shift(-1)

#Show a index for caracters if is more tomorrow
data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

#Show data
data

#Model RandomForest
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

#Training
train = data.iloc[:-100]
test = data.iloc[:-100]

#Data Using
predictors = ['Close', 'Volume', 'Open', 'High', 'Low']
model.fit(train[predictors], train['Target'])
RandomForestClassifier(min_samples_split=100, random_state=1)

#Metrics
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

#Accuracy Test
precision_score(test['Target'], preds)

#Prediction graphs
combined = pd.concat([test['Target'], preds], axis=1)
combined.plot()