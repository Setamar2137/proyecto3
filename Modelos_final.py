#JUAN DAVID CADENA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import save_model

# Cargar el dataframe
df = pd.read_csv('datos_limpios_dummies.csv')

# ---- Modelo 1: Factores socioeconómicos ----
# Variables socioeconómicas
columnas_socioeconomicas = [
    'fami_estratovivienda', 'fami_educacionmadre', 'fami_educacionpadre', 
    'fami_tieneautomovil_Si', 'fami_tienecomputador_Si', 
    'fami_tieneinternet_Si', 'fami_tienelavadora_Si'
]

# Variables de entrada (socioeconómicas)
X1 = df[columnas_socioeconomicas]

# Variables de salida (puntajes de las pruebas)
y1 = df[['punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas', 
         'punt_c_naturales', 'punt_lectura_critica', 'punt_global']]

# Preprocesamiento: Imputación de valores faltantes
# Imputar valores faltantes para columnas numéricas con la media
X1_num = X1.select_dtypes(include=['float64', 'int64'])  # Solo columnas numéricas
X1[X1_num.columns] = X1_num.fillna(X1_num.mean())

# Imputar valores faltantes para columnas categóricas con la moda
X1_cat = X1.select_dtypes(include=['object'])  # Solo columnas categóricas
X1[X1_cat.columns] = X1_cat.apply(lambda x: x.fillna(x.mode()[0]))

# Imputar valores faltantes en y1 (puntajes) con la media
y1 = y1.fillna(y1.mean())  # Para las variables de puntaje

# Convertir las variables categóricas a dummies
X1 = pd.get_dummies(X1, drop_first=True)

# Separar numéricas y dummies
X1_num = X1.select_dtypes(include=['float64', 'int64'])  # Variables numéricas
X1_dummies = X1.select_dtypes(exclude=['float64', 'int64'])  # Variables dummies

# Normalizar las variables numéricas
scaler = StandardScaler()
X1_num_scaled = scaler.fit_transform(X1_num)  # Normalizar las numéricas
# Reconstruir el DataFrame combinado
X1_scaled = pd.DataFrame(X1_num_scaled, columns=X1_num.columns)
X1_final = pd.concat([X1_scaled, X1_dummies.reset_index(drop=True)], axis=1)

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1_final, y1, test_size=0.2, random_state=42)

# Construir el modelo de red neuronal (MLP)
model1 = Sequential()
model1.add(Dense(64, input_dim=X_train1.shape[1], activation='relu'))  # Capa de entrada
model1.add(Dense(32, activation='relu'))  # Capa oculta
model1.add(Dense(y_train1.shape[1], activation='linear'))  # Capa de salida (regresión)

# Compilar el modelo
model1.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history1 = model1.fit(X_train1, y_train1, epochs=100, batch_size=32, validation_data=(X_test1, y_test1))

# Evaluación del modelo
y_pred1 = model1.predict(X_test1)
print(y_pred1[:5])
print(X_test1[:5])
mse1 = mean_squared_error(y_test1, y_pred1)
print(f"Mean Squared Error (MSE) para el Modelo 1: {mse1}")

# Visualizar la pérdida durante el entrenamiento
plt.plot(history1.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history1.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento del Modelo 1')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# ---- Modelo 2: Factores institucionales ----
# Variables institucionales
columnas_institucionales = [
    'cole_bilingue_S', 'cole_calendario', 'cole_caracter', 
    'cole_depto_ubicacion', 'cole_genero', 'cole_jornada'
]

# Variables de entrada (institucionales)
X2 = df[columnas_institucionales]

# Variables de salida (puntajes de las pruebas)
y2 = df[['punt_ingles', 'punt_matematicas', 'punt_sociales_ciudadanas', 
         'punt_c_naturales', 'punt_lectura_critica', 'punt_global']]

# Preprocesamiento: Imputación de valores faltantes
# Imputar valores faltantes para columnas categóricas con la moda
X2_cat = X2.select_dtypes(include=['object'])  # Solo columnas categóricas
X2[X2_cat.columns] = X2_cat.apply(lambda x: x.fillna(x.mode()[0]))

# Imputar valores faltantes en y2 (puntajes) con la media
y2 = y2.fillna(y2.mean())  # Para las variables de puntaje

# Convertir las variables categóricas a dummies
X2 = pd.get_dummies(X2, drop_first=True)


# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Construir el modelo de red neuronal (MLP)
model2 = Sequential()
model2.add(Dense(64, input_dim=X_train2.shape[1], activation='relu'))  # Capa de entrada
model2.add(Dense(32, activation='relu'))  # Capa oculta
model2.add(Dense(y_train2.shape[1], activation='linear'))  # Capa de salida (regresión)

# Compilar el modelo
model2.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo
history2 = model2.fit(X_train2, y_train2, epochs=100, batch_size=32, validation_data=(X_test2, y_test2))

# Evaluación del modelo
y_pred2 = model2.predict(X_test2)
mse2 = mean_squared_error(y_test2, y_pred2)
print(f"Mean Squared Error (MSE) para el Modelo 2: {mse2}")

# Visualizar la pérdida durante el entrenamiento
plt.plot(history2.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history2.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento del Modelo 2')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()


# Guardar el modelo 1
save_model(model1, 'modelo_socioeconomico.h5')
print("Modelo 1 (Factores Socioeconómicos) guardado como 'modelo_socioeconomico.h5'.")
# Guardar el modelo 2
save_model(model2, 'modelo_institucional.h5')
print("Modelo 2 (Factores Institucionales) guardado como 'modelo_institucional.h5'.")