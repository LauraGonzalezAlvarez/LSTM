import matplotlib.pyplot as plt
import numpy as np
np.random.seed(4)
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


# Funciones auxiliares
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real ')
    plt.plot(prediccion, color='blue', label='Predicción ')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()


# Lectura de los datos
dataset = pd.read_csv('datosIndicesSinteticos.csv', index_col='Date', parse_dates=['Date'])
dataset.head()


# Sets de entrenamiento y validación 
set_entrenamiento = dataset[:'2021'].iloc[:,1:2]
set_validacion = dataset['2022':].iloc[:,1:2]

set_entrenamiento['High'].plot(legend=True)
set_validacion['High'].plot(legend=True)
plt.legend(['Entrenamiento (2019-2021)', 'Validación (2022)'])
plt.show()

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 10
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Red LSTM
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 46
modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida, activation='relu'))
modelo.compile(loss='mse',  optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.0035876130103241802), metrics=['accuracy'])
history = modelo.fit(X_train,Y_train,epochs=31,batch_size=32)


# Validación (predicción del valor de las acciones)
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))


prediccion = modelo.predict(X_test)

# Recortar la predicción para que tenga la misma longitud que el conjunto de validación
prediccion = prediccion[:len(set_validacion)]

prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
graficar_predicciones(set_validacion.values,prediccion)

# Graficar la pérdida (loss) en función de las épocas
plt.plot(history.history['loss'])
plt.title('Pérdida en función de las épocas')
plt.ylabel('Pérdida')
plt.xlabel('Épocas')
plt.show()

# Graficar la accuracy (accuracy) en función de las épocas
plt.plot(history.history['accuracy'])
plt.title('Precisión en función de las épocas')
plt.ylabel('Precisión')
plt.xlabel('Épocas')
plt.show()

# Predicciones en el set de validación

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Calcular la diferencia entre las predicciones y los valores reales

set_validacion_ajustado = set_validacion.values[10:]

set_validacion_ajustado = set_validacion_ajustado.reshape(-1)
diferencia = set_validacion_ajustado - prediccion[:, 0]


# Umbral para definir las zonas de compra y venta
umbral = 50

# Generar señales de compra y venta
senal_compra = []
senal_venta = []

for i in range(1, len(diferencia)):
    if diferencia[i] > umbral and diferencia[i-1] < umbral:
        senal_compra.append(i)
    elif diferencia[i] < -umbral and diferencia[i-1] > -umbral:
        senal_venta.append(i)


# Graficar los precios y las señales de compra y venta
plt.plot(set_validacion['High'].values)
plt.plot(senal_compra, set_validacion['High'].values[senal_compra], '^', markersize=10, color='g')
plt.plot(senal_venta, set_validacion['High'].values[senal_venta], 'v', markersize=10, color='r')
plt.title('Señales de trading')
plt.xlabel('Días')
plt.ylabel('Precio de cierre')
plt.legend(['Precio de cierre', 'Señal de compra', 'Señal de venta'])
plt.show()






