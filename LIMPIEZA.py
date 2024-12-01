import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

# Configuración para mostrar más columnas en el dataframe
pd.set_option('display.max_columns', None)

# 1. Cargar el archivo CSV con los datos de Bolívar
df = pd.read_csv('Datos-Bolivar.csv')

# Ver las primeras filas para verificar la carga
print(df.head())

# Seleccionar solo las columnas necesarias
columnas_necesarias = [
    'periodo', 'estu_tipodocumento', 'estu_consecutivo', 'cole_area_ubicacion', 
    'cole_bilingue', 'cole_calendario', 'cole_caracter', 'cole_depto_ubicacion', 
    'cole_genero', 'cole_jornada', 'estu_genero', 
    'fami_educacionmadre', 'fami_educacionpadre', 'fami_estratovivienda', 
    'fami_tieneautomovil', 'fami_tienecomputador', 
    'fami_tieneinternet', 'fami_tienelavadora', 'punt_ingles', 'punt_matematicas', 
    'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global'
]

# Filtrar el dataframe para solo mantener las columnas necesarias
df = df[columnas_necesarias]

# Verificar que las columnas se seleccionaron correctamente
print(df.head())

# Eliminar filas duplicadas
df = df.drop_duplicates()

# Visualización: Gráfico de barras para ver la cantidad de duplicados eliminados
sns.countplot(x=df.duplicated())
plt.title("Datos después de duplicados")
plt.xlabel("Sin Duplicados")
plt.ylabel("Frecuencia")
plt.show()

# Verificar los valores nulos por columna
print("Valores nulos por columna:")
print(df.isnull().sum())

# Imputación para valores faltantes
# Para las variables numéricas, usamos la mediana.
# Para las variables categóricas, usamos la moda.

# Primero, mostramos cómo se verían los valores faltantes antes de imputar.
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Valores nulos antes de imputar")
plt.show()

# Imputación usando SimpleImputer (Mediana para columnas numéricas y Moda para categóricas)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
imputer_numeric = SimpleImputer(strategy='median')
df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])

# Imputar para columnas categóricas
categorical_cols = df.select_dtypes(include=['object']).columns
imputer_categorical = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

# Verificar nuevamente si hay valores nulos
print("Valores nulos después de imputación:")
print(df.isnull().sum())

# Visualización: Gráfico de calor para ver los valores nulos después de la imputación
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Mapa de calor después de imputar")
plt.show()

# Mapeo de categorías
estrato_map = {
    'Estrato 1': 1,
    'Estrato 2': 2,
    'Estrato 3': 3,
    'Estrato 4': 4,
    'Estrato 5': 5,
    'Estrato 6': 6,
    'Sin Estrato': 0,
    'Primaria completa': 0,
    'Educación profesional completa': 0,
    'Ninguno': 0,
    'Secundaria (Bachillerato) incompleta': 0,
    'Primaria incompleta': 0,
    'Secundaria (Bachillerato) completa': 0,
    'Técnica o tecnológica completa': 0,
    'No sabe': 0,
    'Educación profesional incompleta': 0
}

# Aplicar el mapeo a la columna 'fami_estratovivienda'
df['fami_estratovivienda'] = df['fami_estratovivienda'].map(estrato_map)

# Verificar los valores de 'fami_estratovivienda' después del mapeo
print("\nValores únicos de 'fami_estratovivienda' después de la corrección:")
print(df['fami_estratovivienda'].unique())

# Calcular el rango intercuartílico (IQR) para detectar outliers
Q1 = df[['punt_ingles', 'punt_matematicas', 
         'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global']].quantile(0.25)
Q3 = df[['punt_ingles', 'punt_matematicas', 
         'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global']].quantile(0.75)
IQR = Q3 - Q1

# Definir los límites para los outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Eliminar los outliers
df_clean = df[~((df[['punt_ingles', 'punt_matematicas', 
                    'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global']] < limite_inferior) |
                 (df[['punt_ingles', 'punt_matematicas', 
                      'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global']] > limite_superior)).any(axis=1)]

# Verificar la cantidad de datos después de eliminar outliers
print(f"Filas después de eliminar outliers: {df_clean.shape[0]}")

# Visualización: Boxplot para detectar outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[['punt_ingles', 'punt_matematicas', 
                     'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica', 'punt_global']])
plt.title('Boxplot después de outliers')
plt.show()

# Convertir las columnas categóricas en variables dummies
df_clean = pd.get_dummies(df_clean, columns=['cole_bilingue', 'fami_tieneautomovil', 'fami_tienecomputador', 
                                              'fami_tieneinternet', 'fami_tienelavadora'], drop_first=True)

# Eliminar columnas adicionales específicas
df_clean = df_clean.drop(columns=["estu_consecutivo",
                                  "fami_tieneautomovil_Cuatro","fami_tieneautomovil_Dos",
                                  "fami_tieneautomovil_Nueve","fami_tieneautomovil_Ocho",
                                  "fami_tieneautomovil_Once","fami_tieneautomovil_Seis",
                                  "fami_tieneautomovil_Siete","fami_tieneautomovil_Tres",
                                  "fami_tieneautomovil_No"])

# Verificar las primeras filas después de la eliminación
print("Primeras filas después de la eliminación de columnas:")
print(df_clean.head())

#  Guardar el DataFrame limpio (opcional)
df_clean.to_csv('datos_limpios_final.csv', index=False)
