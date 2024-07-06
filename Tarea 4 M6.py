import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Cargar el archivo de datos
data = pd.read_csv(r"E:\Maestria y Diplomados\DIPLOMADO PYTHON\Full info\Modulo 6 aplicaciones de python en machine learning y ciencia de datos\Clase 4\Tarea 4 M6\datos.csv")
# Renombrar las columnas para que sean más fáciles de usar
data.columns = ['TemperaturaPromedio', 'NumeroDeIntegrantes', 'GastoDeLuz_kwh']

# Definir las características (X) y la variable objetivo (y)
X = data[['TemperaturaPromedio', 'NumeroDeIntegrantes']]
y = data['GastoDeLuz_kwh']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer la predicción para una temperatura promedio de 38 grados y 6 miembros de la familia
nueva_temperatura = 38
nuevos_integrantes = 6
prediccion = model.predict(np.array([[nueva_temperatura, nuevos_integrantes]]))

print(f'La predicción del gasto de luz es: {prediccion[0]} kWh')
