import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Crear el Dataset de preuba

np.random.seed(42)
datos = {
    'estacion_id': [f"Estación_{i}" for i in range(1, 16)],
    'afluencia_pasajeros': [120, 115, 130, 20, 15, 30, 70, 80, 75, 140, 25, 85, 10, 125, 65],
    'tiempo_espera_promedio': [2.1, 1.8, 2.5, 8.5, 9.0, 7.5, 4.5, 5.0, 4.2, 2.0, 8.0, 4.8, 10.0, 1.9, 5.5]
}

df = pd.DataFrame(datos)
X = df[['afluencia_pasajeros', 'tiempo_espera_promedio']]

# Escalado de Características

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X)

# Configurar y Entrenar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['categoria_cluster'] = kmeans.fit_predict(X_escalado)

# Visualización 
plt.figure(figsize=(8, 6))

# Mapeo de colores fijos para cada ID de cluster matemático
colores_cluster = {0: 'red', 1: 'blue', 2: 'green'}

# Iteramos sobre cada cluster (0, 1, 2) encontrado por el algoritmo
for cluster_id in range(3):
    componente_cluster = df[df['categoria_cluster'] == cluster_id]
    
    pasajeros_promedio = componente_cluster['afluencia_pasajeros'].mean()
    
    if pasajeros_promedio > 100:
        etiqueta_real = 'Alta Afluencia / Espera Corta'
    elif pasajeros_promedio < 40:
        etiqueta_real = 'Baja Afluencia / Espera Larga'
    else:
        etiqueta_real = 'Afluencia y Espera Media'
       
    # Grafica grupo con su color asignado
    plt.scatter(componente_cluster['afluencia_pasajeros'], 
                componente_cluster['tiempo_espera_promedio'], 
                c=colores_cluster[cluster_id], 
                label=etiqueta_real, 
                s=100, edgecolors='black')

# Detalles estéticos de la gráfica
plt.title('Segmentación No Supervisada de Estaciones de Transporte')
plt.xlabel('Afluencia de Pasajeros (personas/min)')
plt.ylabel('Tiempo de Espera Promedio (min)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
