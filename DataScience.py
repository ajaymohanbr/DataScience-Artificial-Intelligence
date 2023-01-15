import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Observe Database
table = pd.read_csv('advertising.csv')

# Calculate the correlation:
print(table.corr())

# Demonstrate the Corr in a graph

# Create graph
sns.heatmap(table.corr(), cmap="Greens", annot=True)
# Show graph
plt.show()

# Creating AI

# 1: Create x and y
y = table['Vendas']
x = table[['TV', 'Radio', 'Jornal']]

# 2. Training datas and Test datas.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Selected 2 models of AI: Regression Linear and Random Forest

# Create models of AI:
regressionlinear_model = LinearRegression()
randomforest_model = RandomForestRegressor()

# Train the models:
regressionlinear_model.fit(x_train, y_train)
randomforest_model.fit(x_train, y_train)

# Test the models
prediction_regressionlinear = regressionlinear_model.predict(x_test)
prediction_randomforest = randomforest_model.predict(x_test)

# Compare the 2 models:
print(r2_score(y_test, prediction_regressionlinear))
print(r2_score(y_test, prediction_randomforest))

# To view the prediction in graphic

table_assit = pd.DataFrame()
table_assit['y_test'] = y_test
table_assit['prediction Linear'] = prediction_regressionlinear
table_assit['prediction Random'] = prediction_randomforest

sns.lineplot(data=table_assit)
plt.show()

# New Prediction

newtable = pd.read_csv('novos.csv')
print(newtable)
prediction_new = randomforest_model.predict(newtable)
print(prediction_new)
