import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv('housing.csv',header=None, delimiter=r"\s+", names=column_names)
print(dataset.head(6))
data = pd.DataFrame(dataset)
data.columns = column_names
print(data.head(5))
data.describe()
corr = data.corr()
corr.shape
plot.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.2f', annot=True, annot_kws={'size':15}, cmap='Blues')
X = data.drop(['MEDV'], axis = 1)
Y = data['MEDV']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 2)
# Импортируем линейную регрессию
from sklearn.linear_model import LinearRegression
# Создаем объект линейной регрессии
lm = LinearRegression()

# Обучаем модель, используя "тренировочные" данные
lm.fit(X_train, Y_train)
# Значение коээфициента b (y=ax+b)
lm.intercept_
coefficients = pd.DataFrame([X_train.columns,lm.coef_]).T
coefficients = coefficients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
print(coefficients)
# Предсказание цены тренировочной части данных
Y_pred = lm.predict(X_train)
# Оцениваем результаты точности модели
from sklearn import metrics
print('R^2:',metrics.r2_score(Y_train, Y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_train, Y_pred))*(len(Y_train)-1)/(len(Y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_train, Y_pred))
print('MSE:',metrics.mean_squared_error(Y_train, Y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_train, Y_pred)))
# Визуализируем разность между фактической цены и предсказанной
plot.scatter(Y_train, Y_pred)
plot.xlabel("Prices")
plot.ylabel("Predicted prices")
plot.title("Prices vs Predicted prices")
plot.show()
# Проверка остатков
plot.scatter(Y_pred,Y_pred-Y_train)
plot.title("Predicted vs residuals")
plot.xlabel("Predicted")
plot.ylabel("Residuals")
plot.show()
# Предсказание цены тестовой части данных
Y_test_pred = lm.predict(X_test)
# Оценка модели
R2 = metrics.r2_score(Y_test, Y_test_pred)
print('R^2:', R2)
print('Adjusted R^2:',1 - (1-metrics.r2_score(Y_test, Y_test_pred))*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(Y_test, Y_test_pred))
print('MSE:',metrics.mean_squared_error(Y_test, Y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))
# Визуализируем разность между фактической цены и предсказанной
plot.scatter(Y_test, Y_test_pred)
plot.xlabel("Prices")
plot.ylabel("Predicted prices")
plot.title("Prices vs Predicted prices")
plot.show()
# Проверка результатов
plot.scatter(Y_test_pred,Y_test-Y_test_pred)
plot.title("Predicted vs residuals")
plot.xlabel("Predicted")
plot.ylabel("Residuals")
plot.show()
Y_bin = np.zeros_like(Y_test)
threshold = data['MEDV'].mean()
Y_bin[Y_test>=threshold] = 1

fpr, tpr, thresholds = metrics.roc_curve(Y_bin,  Y_test_pred)
print(thresholds)
print(Y_test_pred.size)
print(Y_bin.size)
#create ROC curve
plot.plot(fpr,tpr, color="green")
plot.ylabel('True Positive Rate', color="red")
plot.xlabel('False Positive Rate', color="red")
plot.show()
