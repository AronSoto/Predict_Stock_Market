#Importing all complements
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

#Importing data in the country context(Perú)
data = pd.read_csv('../Data/Data_Intercorp.csv')

#Showing data a company
print(data)

#Show the data visually (Gráficas)
data['Close'].plot()

#Split the data into training and testing data sets (Entrenamiento)
train_data = data.iloc[:int(.99*len(data)), :]
test_data = data.iloc[:int(.99*len(data)), :]

#Define the feature and target variable (Definición)
features = ['Open','Volume']
target = 'Close'

#Create and train the model (Modelación)
model = xgb.XGBRegressor()
model.fit(train_data[features], train_data[target])

#Make and show the predictions on the test data (Muestra)
predictions = model.predict(test_data[features])
print('Model Predictions Intercorp: ')
print(predictions)

#Show the actual values (Mostrar)
print('Actual Values: ')
print(test_data[target])

#Show the models accuracy (Exactitud)
accuracy = model.score(test_data[features], test_data[target])
print('Accuracy: ')
print(accuracy)

#Plot the predictions and the close price (Trazamiento Graficas)
plt.plot(data['Close'], label = 'Close Price')
plt.plot(test_data[target].index, predictions, label = 'Predictions')
plt.legend()
plt.show()