#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import pickle
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')




# In[2]:


df = pd.read_csv('../data/datos_procesados.csv')
df.head()


# In[3]:


df.columns


# In[4]:


# X = df.drop('Precio_log', axis=1)
# # Normalizo 
# scaler_x = MinMaxScaler()
# scaler_x.fit(X)
# X = scaler_x.transform(X)

# y = df['Precio_log']

# X.shape, y.shape


# In[5]:


# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # In[6]:


# from sklearn.ensemble import  RandomForestRegressor

# rf_1 = RandomForestRegressor()
# rf_1.get_params()


# In[7]:


# rf_1.fit(X_train, y_train)


# In[8]:


# Check training
# y_pred_train = rf_1.predict(X_train)

# rmse
# from sklearn.metrics import  mean_squared_error

# rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

# print('RMSE Train data {}'.format(rmse_train))


# In[9]:


# error test data
# y_pred_test = rf_1.predict(X_test)

# rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# print('RMSE Test data {}'.format(rmse_test))


# In[10]:


# Check puntuacion traning data.
# from sklearn.metrics import r2_score

# y_pred_train = rf_1.predict(X_train)

# print('Puntuacion entrenamiento: {}'.format(r2_score(y_train, y_pred_train)))


# In[11]:


# y_pred_test = rf_1.predict(X_test)
# print('Puntuacion Test: {}'.format(r2_score(y_test, y_pred_test)))


# In[12]:


# plotting train
# check prediccion con valores originales
# plt.figure(figsize=(17,7))
# y_pred_train = rf_1.predict(X_train)
# plt.subplot(1,2,1)
# plt.scatter(y_train, y_pred_train, alpha=0.2);
# plt.xlabel('Objetivo (y_train)');
# plt.ylabel('Prediccion (y_pred)');
# plt.title('Tiempo de train');

# plotting test error
# y_pred_test = rf_1.predict(X_test)
# plt.subplot(1,2,2)
# plt.scatter(y_test, y_pred_test, alpha=0.2);
# plt.xlabel('Objetivos (y_test)');
# plt.ylabel('Prediccion (y_pred_train)');
# plt.title('Tiempo test');


# In[13]:


# plotting distribucion residual
# residual_train = (y_train - rf_1.predict(X_train))
# residual_test = (y_test - rf_1.predict(X_test))

# # ploting en el entrenamiento
# plt.figure(figsize=(17,7))
# plt.subplot(1,2,1)
# sns.distplot(residual_train)
# plt.title('Traning Residual PDF')

# # ploting test
# plt.subplot(1,2,2)
# sns.distplot(residual_test)
# plt.title('Testting Residual PDF')


# In[14]:


# Crear df comparar resultados y prediccion 
# df_eval = pd.DataFrame(rf_1.predict(X_test), columns=['Prediccion'])
# # adding column
# y_test = y_test.reset_index(drop=True)
# df_eval['Objetivo'] = y_test

# # creando columnas residual y difference
# df_eval['Residual'] = df_eval['Objetivo'] - df_eval['Prediccion']
# df_eval['Difference%'] = np.absolute(df_eval['Residual'] / df_eval['Objetivo']*100)
# # check 
# df_eval


# In[15]:


# df_eval.describe()


# In[16]:


#Visualizar un indicador del tipo oscilador
# X = np.linspace(0, 250, 250)
# Y1 = df_eval["Prediccion"].head(250).values
# Y2 = df_eval["Objetivo"].head(250).values

# plt.figure(figsize = (25, 8))
# # Plot de X**2
# plt.plot(X, Y1, color="blue")
# # Plot de X**3

# plt.plot(X, Y2,color="red")


# plt.show()


# In[17]:


# from sklearn.metrics import mean_absolute_error
# n = len(y_test)
# k = len(y_pred_test)
# r_cuadrado_ajustado = 1 - ((1 - r2_score(y_test, y_pred_test)) * (n - 1) / (n - k - 1))
# def calcular_metricas():
#     # Cálculo de métricas
#     metrics = {
#         'Error cuadrático medio (RMSE)': rmse_train,
#         'r2_score': r2_score(y_test, y_pred_test),
#         'r2_score_adjusted': r_cuadrado_ajustado,
#         'Error absoluto medio (MAE)': mean_absolute_error(y_test, y_pred_test)
        
#     }
#     return metrics


# In[ ]:


# Para guardar el modelo

# joblib.dump(rf_1, 'randomforest.pkl', compress=3)


# In[18]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def generar_grafica(X_train, y_train, X_test, y_test):
    # rf_1 = RandomForestRegressor()
    # rf_1.fit(X_train, y_train)

    rf_1 = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))

    # plotting train
    y_pred_train = rf_1.predict(X_train)
    plt.figure(figsize=(17, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.2)
    plt.xlabel('Objetivo (y_train)')
    plt.ylabel('Predicción (y_pred)')
    plt.title('Tiempo de entrenamiento')

    # plotting test error
    y_pred_test = rf_1.predict(X_test)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.2)
    plt.xlabel('Objetivos (y_test)')
    plt.ylabel('Predicción (y_pred_train)')
    plt.title('Tiempo de test')

    return plt


# In[1]:


def distribucion_residual(X_train, y_train, X_test, y_test):
    # Inicializa el modelo RandomForestRegressor
    # rf_model = RandomForestRegressor()
    
    # Entrena el modelo
    # rf_model.fit(X_train, y_train)
    rf_model = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))
    
    # Calcula los residuos para los datos de entrenamiento y prueba
    residual_train = y_train - rf_model.predict(X_train)
    residual_test = y_test - rf_model.predict(X_test)

    # Plotting para el conjunto de entrenamiento
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    sns.histplot(residual_train, kde=True)  # Utiliza histplot en lugar de distplot
    plt.title('Residuos del Entrenamiento')

    # Plotting para el conjunto de prueba
    plt.subplot(1,2,2)
    sns.histplot(residual_test, kde=True)  # Utiliza histplot en lugar de distplot
    plt.title('Residuos del Test')

    # Devuelve el gráfico
    return plt


# In[2]:


def graf_oscilador(X_test, y_test):

    rf_1 = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))

    # Crear DataFrame para comparar resultados y predicciones
    df_eval = pd.DataFrame(rf_1.predict(X_test), columns=['Prediccion'])
    # Añadir columna 'Objetivo'
    y_test = y_test.reset_index(drop=True)
    df_eval['Objetivo'] = y_test

    # Calcular columnas 'Residual' y 'Difference%'
    df_eval['Residual'] = df_eval['Objetivo'] - df_eval['Prediccion']
    df_eval['Difference%'] = np.absolute(df_eval['Residual'] / df_eval['Objetivo'] * 100)

    # Visualizar un indicador tipo oscilador
    # X = np.linspace(0, 250, 250)
    # Y1 = df_eval["Prediccion"].head(250).values
    # Y2 = df_eval["Objetivo"].head(250).values

    # plt.figure(figsize=(25, 8))
    # plt.plot(X, Y1, color="blue", label="Predicción")
    # plt.plot(X, Y2, color="red", label="Objetivo")
    # plt.legend()
    # plt.title('Gráfico Oscilador de Predicciones y Objetivos')
    # plt.xlabel('Índice')
    # plt.ylabel('Valores')
    # plt.show()
    X = np.linspace(0, 150, 150)
    Y1 = df_eval["Prediccion"].head(150).values
    Y2 = df_eval["Objetivo"].head(150).values

    valores = np.concatenate((Y1,Y2))


    indice1 =np.full(Y1.shape[0], fill_value="Predicción")
    indice2 =np.full(Y2.shape[0], fill_value="Valor real")
    indices = np.concatenate((indice1,indice2))

    df_comparando = pd.DataFrame({'x': np.concatenate((X,X)),
                    'y': valores,
                    'grupo' : indices})

    fig =  px.line(df_comparando, x = 'x', y = 'y', color = 'grupo',color_discrete_map = {'Predicción':'blue', 'Valorr real':'red'})

    fig.update_layout(
        autosize=False,
        width=1400,
        height=600,
    )
    return fig