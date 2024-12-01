from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

# Cargar los datos y reducir su tamaño # LO REDUJE POR MI PC, no se si en la red funcione con todos
data = pd.read_csv("datos_limpios_dummies.csv")
data_reduced = data.sample(frac=0.5, random_state=42) #cambiar frac valores entre 0 y 1 para reducir el numero de datos

# Cargar los modelos
model_socioeconomico = load_model('modelo_socioeconomico.h5')
model_institucional = load_model('modelo_institucional.h5')

# Crear lista de columnas de puntajes
puntajes = ['punt_global', 'punt_ingles', 'punt_matematicas', 
            'punt_sociales_ciudadanas', 'punt_c_naturales', 'punt_lectura_critica']

# Variables de entrada para cada modelo
socio_variables = {
    'fami_estratovivienda': sorted(data['fami_estratovivienda'].unique()),
    'fami_educacionmadre': sorted(data['fami_educacionmadre'].unique()),  # Eliminar primera categoría
    'fami_educacionpadre': sorted(data['fami_educacionpadre'].unique()),
    'fami_tieneautomovil_Si': [0, 1],
    'fami_tienecomputador_Si': [0, 1],
    'fami_tieneinternet_Si': [0, 1],
    'fami_tienelavadora_Si': [0, 1]
}
inst_variables = {
    'cole_area_ubicacion': sorted(data['cole_area_ubicacion'].unique()),
    'cole_calendario': sorted(data['cole_calendario'].unique()),
    'cole_caracter': sorted(data['cole_caracter'].unique()),
    'cole_genero': sorted(data['cole_genero'].unique()),
    'cole_jornada': sorted(data['cole_jornada'].unique())
}

# Crear la aplicación Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Análisis del Desempeño en las Pruebas Saber 11"),
    
    dcc.Tabs([
        # Tab de Análisis Socioeconómico
        dcc.Tab(label="Análisis Socioeconómico", children=[
            html.Div([
                html.H3("Filtros"),
                html.Label("Selecciona el Tipo de Puntaje:"),
                dcc.Dropdown(
                    id='socio-puntaje',
                    options=[{'label': p, 'value': p} for p in puntajes],
                    value='punt_global'
                ),
                
                html.H3("Gráficas"),
                dcc.Graph(id='violin-madre'),
                dcc.Graph(id='violin-padre'),
                dcc.Graph(id='barras-estrato'),
                dcc.Graph(id='violin-bienes'),
                # Modelo de Predicción (Factores Socioeconómicos)
                html.H3("Modelo de Predicción (Factores Socioeconómicos)"),
                html.Div([
                    *[
                        html.Div([
                            html.Label(var),
                            dcc.Dropdown(
                                id=f'dropdown-{var}',
                                options=[{'label': str(v), 'value': v} for v in values],
                                value=values[0]
                            )
                        ]) for var, values in socio_variables.items()
                    ],
                    html.Div(id='socio-model-output', style={'whiteSpace': 'pre-line'})
                ])
            ])
        ]),

        # Tab de Análisis de Factores Institucionales
        dcc.Tab(label="Análisis de Factores Institucionales", children=[
            html.Div([
                html.H3("Filtros"),
                html.Label("Selecciona el Tipo de Puntaje:"),
                dcc.Dropdown(
                    id='inst-puntaje',
                    options=[{'label': p, 'value': p} for p in puntajes],
                    value='punt_global'
                ),
                
                html.H3("Gráficas"),
                dcc.Graph(id='violin-area'),
                dcc.Graph(id='barras-calendario'),
                dcc.Graph(id='violin-caracter'),
                dcc.Graph(id='lineas-genero'),
                dcc.Graph(id='violin-jornada'),
                # Modelo de Predicción (Factores Institucionales)
                html.H3("Modelo de Predicción (Factores Institucionales)"),
                html.Div([
                    *[
                        html.Div([
                            html.Label(var),
                            dcc.Dropdown(
                                id=f'dropdown-{var}',
                                options=[{'label': str(v), 'value': v} for v in values],
                                value=values[0]
                            )
                        ]) for var, values in inst_variables.items()
                    ],
                    html.Div(id='inst-model-output', style={'whiteSpace': 'pre-line'})
                ])
            ])
        ])
    ])
])

# Callbacks para Análisis Socioeconómico
@app.callback(
    [Output('violin-madre', 'figure'),
     Output('violin-padre', 'figure'),
     Output('barras-estrato', 'figure'),
     Output('violin-bienes', 'figure')],
    [Input('socio-puntaje', 'value')]
)
def update_socio_graphs(selected_puntaje):
    # Violin Plot: Distribución según educación de la madre
    violin_madre = px.violin(
        data_reduced, x='fami_educacionmadre', y=selected_puntaje, box=True,
        title=f'Distribución de {selected_puntaje} según Educación de la Madre',
        labels={'fami_educacionmadre': 'Educación de la Madre'},
        color='fami_educacionmadre'
    ).update_traces(points=False).update_xaxes(categoryorder="total descending")
    
    # Violin Plot: Distribución según educación del padre
    violin_padre = px.violin(
        data_reduced, x='fami_educacionpadre', y=selected_puntaje, box=True,
        title=f'Distribución de {selected_puntaje} según Educación del Padre',
        labels={'fami_educacionpadre': 'Educación del Padre'},
        color='fami_educacionpadre'
    ).update_traces(points=False).update_xaxes(categoryorder="total descending")
    
    # Gráfico de Barras: Puntajes por estrato
    barras_estrato = px.bar(
        data_reduced.groupby('fami_estratovivienda')[selected_puntaje].mean().reset_index(),
        x='fami_estratovivienda', y=selected_puntaje,
        title=f'Promedio de {selected_puntaje} por Estrato',
        labels={'fami_estratovivienda': 'Estrato'},
        color='fami_estratovivienda'
    )
    
    # Violin Plot: Distribución según bienes del hogar
    bienes_df = data_reduced.melt(
        id_vars=[selected_puntaje],
        value_vars=['fami_tieneautomovil_Si', 'fami_tienecomputador_Si', 
                    'fami_tieneinternet_Si', 'fami_tienelavadora_Si'],
        var_name='Bien', value_name='Tiene'
    )
    bienes_df['Bien'] = bienes_df['Bien'].str.replace('fami_tiene', '').str.replace('_Si', '').str.capitalize()
    bienes_df['Tiene'] = bienes_df['Tiene'].replace({1: 'Sí', 0: 'No'})
    
    violin_bienes = px.violin(
        bienes_df, x='Bien', y=selected_puntaje, color='Tiene', box=True,
        title=f'Distribución de {selected_puntaje} según Bienes del Hogar',
        labels={'Bien': 'Bien del Hogar', 'Tiene': '¿Posee el Bien?'}
    ).update_traces(points=False)
    
    return violin_madre, violin_padre, barras_estrato, violin_bienes

# Función para transformar las entradas categóricas en One-Hot Encoding
def transform_input(values, variables, expected_shape):
    input_data = []
    for val, (var, categories) in zip(values, variables.items()):
        if isinstance(categories[0], str):  # Si la variable es categórica
            one_hot = [1 if val == cat else 0 for cat in categories]
            input_data.extend(one_hot)
        else:  # Si la variable es numérica
            input_data.append(val)
    # Asegurar que la entrada tenga el tamaño esperado
    input_data = np.array(input_data).reshape(1, -1)
    if input_data.shape[1] > expected_shape:
        input_data = input_data[:, :expected_shape]  # Truncar si hay columnas extras
    elif input_data.shape[1] < expected_shape:
        padding = np.zeros((1, expected_shape - input_data.shape[1]))  # Completar con ceros si faltan columnas
        input_data = np.hstack([input_data, padding])
    return input_data

# Callback para predicción del modelo socioeconómico
@app.callback(
    Output('socio-model-output', 'children'),
    [Input(f'dropdown-{var}', 'value') for var in socio_variables]
)
def predict_socio(*values):
    input_data = transform_input(values, socio_variables, expected_shape=27) 
    prediction = model_socioeconomico.predict(input_data)
    puntaje_global = prediction[0][-1]  # Último valor corresponde al puntaje global
    return f"Predicción del Puntaje Global (Socioeconómico): {puntaje_global:.2f}"


# Callbacks para Análisis de Factores Institucionales
@app.callback(
    [Output('violin-area', 'figure'),
     Output('barras-calendario', 'figure'),
     Output('violin-caracter', 'figure'),
     Output('lineas-genero', 'figure'),
     Output('violin-jornada', 'figure')],
    [Input('inst-puntaje', 'value')]
)
def update_inst_graphs(selected_puntaje):
    # Gráfico de Violín: Área de Ubicación
    violin_area = px.violin(
        data_reduced, x='cole_area_ubicacion', y=selected_puntaje, box=True,
        title=f'Distribución de {selected_puntaje} según Área de Ubicación',
        labels={'cole_area_ubicacion': 'Área de Ubicación'},
        color='cole_area_ubicacion'
    ).update_traces(points=False)
    
    # Gráfico de Barras: Calendario
    barras_calendario = px.bar(
        data_reduced.groupby('cole_calendario')[selected_puntaje].mean().reset_index(),
        x='cole_calendario', y=selected_puntaje,
        title=f'Promedio de {selected_puntaje} por Calendario',
        labels={'cole_calendario': 'Calendario'},
        color='cole_calendario'
    )
    
    # Gráfico de Violín: Carácter del Colegio
    violin_caracter = px.violin(
        data_reduced, x='cole_caracter', y=selected_puntaje, box=True,
        title=f'Distribución de {selected_puntaje} según Carácter del Colegio',
        labels={'cole_caracter': 'Carácter del Colegio'},
        color='cole_caracter'
    ).update_traces(points=False)
    
    # Gráfico de Líneas: Género del Colegio
    lineas_genero = px.line(
        data_reduced.groupby('cole_genero')[selected_puntaje].mean().reset_index(),
        x='cole_genero', y=selected_puntaje,
        title=f'Tendencia de {selected_puntaje} por Género del Colegio',
        labels={'cole_genero': 'Género del Colegio'}
    )
    
    # Gráfico de Violín: Jornada
    violin_jornada = px.violin(
        data_reduced, x='cole_jornada', y=selected_puntaje, box=True,
        title=f'Distribución de {selected_puntaje} según Jornada',
        labels={'cole_jornada': 'Jornada'},
        color='cole_jornada'
    ).update_traces(points=False)
    
    return violin_area, barras_calendario, violin_caracter, lineas_genero, violin_jornada

# Callback para predicción del modelo institucional
@app.callback(
    Output('inst-model-output', 'children'),
    [Input(f'dropdown-{var}', 'value') for var in inst_variables]
)
def predict_inst(*values):
    input_data = transform_input(values, inst_variables, expected_shape=13)  
    prediction = model_institucional.predict(input_data)
    puntaje_global = prediction[0][-1]  # Último valor corresponde al puntaje global
    return f"Predicción del Puntaje Global (Institucional): {puntaje_global:.2f}"

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
