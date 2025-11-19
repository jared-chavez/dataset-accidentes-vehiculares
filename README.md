# Dataset de Accidentes Movilísticos

Este repositorio contiene un dataset limpio de accidentes movilísticos generado a partir de datos brutos mediante un proceso automatizado de limpieza y validación de datos. El dataset está diseñado para análisis de extracción de conocimientos y minería de datos.

## Estructura del Repositorio

```
dataset/
├── raw_accidents_data.csv                    # Dataset original (datos brutos)
├── cleaned_accidents_data.csv                # Dataset limpio (procesado)
├── supervised_learning.py                    # Script principal de ML supervisado (wrapper)
├── unsupervised_learning.py                  # Script principal de ML no supervisado (wrapper)
├── data_cleaning_script.py                   # Script de limpieza automatizada
├── advanced_etl_tool.py                     # Herramienta ETL avanzada
├── requirements.txt                          # Dependencias de Python
├── venv/                                     # Entorno virtual (compartido)
├── .gitignore                                # Archivos a ignorar en Git
├── README.md                                 # Este archivo
├── supervised_learning/                     # Módulo de análisis supervisado
│   ├── supervised_learning_main.py          # Script principal
│   ├── models/                               # Modelos ML
│   ├── preprocessing/                       # Preparación de datos
│   ├── evaluation/                           # Evaluación y métricas
│   └── results/                              # Resultados generados
└── unsupervised_learning/                   # Módulo de análisis no supervisado
    ├── unsupervised_learning_main.py         # Script principal
    ├── models/                               # Modelos de clustering y reducción
    ├── preprocessing/                         # Preparación de datos
    ├── evaluation/                           # Evaluación y métricas
    ├── optimization/                          # Optimización de hiperparámetros
    └── results/                              # Resultados generados
└── data/                                     # Carpeta con archivos adicionales
    ├── datawarehouse_factaccidents.csv       # Tabla de hechos del data warehouse
    ├── datawarehouse_dimtime.csv             # Dimensión tiempo
    ├── datawarehouse_dimlocation.csv         # Dimensión ubicación
    ├── datawarehouse_dimvehicle.csv          # Dimensión vehículo
    ├── datawarehouse_dimdriver.csv           # Dimensión conductor
    └── unified_dimensions_table.csv          # Tabla de dimensiones unificada
```

## Especificaciones del Dataset

### Dataset Limpio (`cleaned_accidents_data.csv`)

**Registros:** 2363 accidentes  
**Período:** Enero 2021 - Diciembre 2024  
**Calidad:** 77.2% de datos conservados (697 registros eliminados por problemas de calidad)

#### Columnas del Dataset

| Columna | Tipo | Descripción | Valores Posibles |
|---------|------|-------------|------------------|
| `incident_id` | Entero | Identificador único del accidente | 101-3160 |
| `incident_date` | Fecha | Fecha del accidente | Formato: YYYY-MM-DD |
| `driver_age` | Decimal | Edad del conductor | 18-80 años |
| `road_conditions` | Categórico | Condiciones del camino | Dry, Wet, Icy |
| `weather_conditions` | Categórico | Condiciones climáticas | Sunny, Rain, Snow, Cloudy, Clear |
| `accident_severity` | Categórico | Severidad del accidente | Minor, Serious, Critical |
| `number_of_vehicles` | Entero | Número de vehículos involucrados | 1-8 |
| `number_of_fatalities` | Entero | Número de fatalidades | 0-5 |

#### Estadísticas del Dataset

- **Rango de fechas:** 2021-01-02 a 2024-12-31
- **Edad promedio de conductores:** 51.4 años
- **Total de vehículos involucrados:** 13105
- **Total de fatalidades:** 8865

#### Distribución por Severidad

- **Minor:** 1035 accidentes (35.2%)
- **Critical:** 961 accidentes (32.7%)
- **Serious:** 942 accidentes (32.1%)

## Data Warehouse - Tablas Dimensionales

El proyecto incluye un diseño de data warehouse basado en el esquema de estrella, con una tabla de hechos central y tablas de dimensión que proporcionan contexto descriptivo.

### Tabla de Hechos (FactAccidents)
- **Archivo:** `datawarehouse_factaccidents.csv`
- **Contenido:** Métricas cuantitativas y claves foráneas
- **Campos:** accident_key, date_key, location_key, vehicle_key, driver_key, number_of_vehicles, number_of_fatalities, accident_severity

### Tablas de Dimensión

#### DimTime (Dimensión Tiempo)
- **Archivo:** `datawarehouse_dimtime.csv`
- **Campos:** date_key, incident_date, day_of_week, month, year
- **Propósito:** Análisis temporal de accidentes

#### DimLocation (Dimensión Ubicación)
- **Archivo:** `datawarehouse_dimlocation.csv`
- **Campos:** location_key, road_conditions, weather_conditions, track
- **Propósito:** Análisis por condiciones ambientales y rutas (track) donde ocurren accidentes

#### DimVehicle (Dimensión Vehículo)
- **Archivo:** `datawarehouse_dimvehicle.csv`
- **Campos:** vehicle_key, vehicle_type, vehicle_make, vehicle_model
- **Propósito:** Análisis por características del vehículo

#### DimDriver (Dimensión Conductor)
- **Archivo:** `datawarehouse_dimdriver.csv`
- **Campos:** driver_key, driver_age, driver_gender
- **Propósito:** Análisis demográfico de conductores

## Herramienta ETL Avanzada

### Características Principales

La herramienta ETL avanzada (`advanced_etl_tool.py`) soporta múltiples formatos de entrada y salida:

#### Formatos de Entrada Soportados
- **CSV** - 
- **Excel** - 
- **JSON** - 
- **SQL** - 

#### Formatos de Salida Soportados
- **CSV** - Para análisis en Python/R
- **Excel** - Para análisis en Excel
- **JSON** - Para aplicaciones web
- **SQL** - Base de datos SQLite
- **Power BI** - Archivos optimizados para Power BI

#### Configuración de Limpieza Ajustable
- **75% de conservación** - Configurado para mantener 3/4 de los datos
- **Limpieza por etapas** - Fechas, edades, categóricos, numéricos
- **Validación automática** - Detección y corrección de problemas

### Tabla de Dimensiones Unificada

El archivo `unified_dimensions_table.csv` contiene todas las dimensiones en una sola tabla:

- **Dimensiones temporales**: día, mes, año, trimestre, fin de semana
- **Dimensiones del conductor**: edad, género, grupo etario
- **Dimensiones del vehículo**: tipo, marca, modelo
- **Dimensiones ambientales**: condiciones del camino y clima
- **Dimensiones estacionales**: primavera, verano, otoño, invierno

## Proceso de Limpieza

### Problemas Identificados en los Datos Originales

1. **Fechas inconsistentes:** Múltiples formatos (YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY)
2. **Valores faltantes:** Edades con `NaN`
3. **Inconsistencias en capitalización:** `wet` vs `Wet`, `rain` vs `Rain`
4. **Valores atípicos:** Edad de 110 años (eliminado)
5. **Inconsistencias en severidad:** `Minor` vs `minor`

### Transformaciones Aplicadas

1. **Estandarización de fechas:** Todas convertidas a formato YYYY-MM-DD
2. **Limpieza de edades:** 
   - Eliminación de valores atípicos (< 16 o > 100 años)
   - Imputación de valores faltantes usando mediana por severidad
3. **Estandarización categórica:** Capitalización consistente
4. **Validación numérica:** Verificación de rangos válidos
5. **Eliminación de duplicados:** No se encontraron duplicados

## Cómo Generar el Dataset Limpio

### Prerrequisitos

- Python 3.7+
- pip (gestor de paquetes de Python)

### Instalación y Ejecución

1. **Clonar o descargar el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   cd dataset
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el script de limpieza:**
   ```bash
   python data_cleaning_script.py
   ```

### Salida Esperada

El script generará:
- `cleaned_accidents_data.csv`: Dataset limpio
- Reporte de calidad de datos en consola
- Estadísticas del dataset procesado

## Análisis de Machine Learning

El proyecto incluye dos módulos completos de análisis de machine learning: **supervisado** y **no supervisado**.

## Análisis Supervisado - Machine Learning

El módulo de análisis supervisado permite entrenar y evaluar modelos de clasificación y regresión.

### Estructura del Módulo

```
supervised_learning/
├── supervised_learning_main.py      # Script principal
├── models/
│   ├── classification_models.py      # Modelos de clasificación
│   ├── regression_models.py          # Modelos de regresión
│   └── model_utils.py                # Utilidades
├── preprocessing/
│   ├── data_preparation.py           # Preparación de datos
│   └── feature_engineering.py        # Feature engineering
├── evaluation/
│   ├── metrics_calculation.py       # Cálculo de métricas
│   └── visualizations.py             # Visualizaciones
└── results/
    ├── classification_results/       # Resultados de clasificación
    └── regression_results/           # Resultados de regresión
```

### Problemas Supervisados

#### 1. Clasificación - Predicción de Severidad del Accidente

**Tipo:** Clasificación multiclase (3 clases)  
**Variable Objetivo:** `accident_severity` (Minor, Serious, Critical)  
**Algoritmos Implementados:**
- Logistic Regression (baseline)
- Random Forest Classifier
- Gradient Boosting (XGBoost) - opcional

**Métricas de Evaluación:**
- Accuracy, Precision, Recall, F1-Score (Macro y Weighted)
- Matriz de confusión
- Importancia de features

#### 2. Regresión - Predicción de Número de Fatalidades

**Tipo:** Regresión (valor numérico continuo)  
**Variable Objetivo:** `number_of_fatalities` (0-5)  
**Algoritmos Implementados:**
- Linear Regression (baseline)
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor (XGBoost) - opcional

**Métricas de Evaluación:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de Determinación)
- MAPE (Mean Absolute Percentage Error)

### Ejecución del Análisis Supervisado

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar desde la raíz del proyecto:**
   ```bash
   python supervised_learning.py
   ```

   O desde dentro de la carpeta:
   ```bash
   cd supervised_learning
   python supervised_learning_main.py
   ```

### Proceso de Entrenamiento

1. **Preparación de Datos:**
   - Carga y merge de datasets (`cleaned_accidents_data.csv` + `unified_dimensions_table.csv`)
   - Feature engineering (creación de variables derivadas)
   - Encoding de variables categóricas (One-Hot Encoding)
   - Normalización de features numéricas

2. **División de Datos:**
   - Train: 70% de los datos
   - Validation: 15% de los datos
   - Test: 15% de los datos
   - Estratificación para mantener distribución de clases

3. **Entrenamiento:**
   - Entrenamiento de múltiples modelos
   - Optimización de hiperparámetros (opcional)
   - Validación en conjunto de validación

4. **Evaluación:**
   - Evaluación en conjunto de test
   - Cálculo de métricas completas
   - Generación de visualizaciones

### Resultados Generados

**Clasificación:**
- `metrics_table.csv` - Tabla comparativa de métricas
- `confusion_matrix.png` - Matriz de confusión
- `metrics_comparison.png` - Comparación de métricas
- `feature_importance.png` - Importancia de features
- `feature_importance.csv` - Datos de importancia

**Regresión:**
- `metrics_table.csv` - Tabla comparativa de métricas
- `predictions_vs_real.png` - Predicciones vs valores reales
- `residuals_plot.png` - Análisis de residuales
- `metrics_comparison.png` - Comparación de métricas
- `feature_importance.png` - Importancia de features
- `feature_importance.csv` - Datos de importancia

### Feature Engineering

El módulo crea automáticamente features derivadas:
- Features temporales: día del mes, semana del año, inicio/fin de mes
- Features de edad: edad al cuadrado, indicadores de senior/joven
- Features de interacción: fatalidades por vehículo, indicadores de alta severidad
- Features de condiciones: combinación de condiciones del camino y clima

### Casos de Uso para Extracción de Conocimientos

### Análisis Exploratorio de Datos (EDA)
- Distribución temporal de accidentes
- Análisis de factores de riesgo (edad, condiciones climáticas)
- Correlaciones entre variables

### Minería de Datos
- Clasificación de severidad de accidentes
- Agrupación de accidentes por patrones similares
- Predicción de factores de riesgo

### Visualizaciones Sugeridas
- Gráficos de barras por severidad y condiciones
- Heatmaps de correlación
- Series temporales de accidentes
- Diagramas de dispersión edad vs severidad

## Ejemplos de Consultas

### Análisis de Accidentes por Condiciones Climáticas
```python
import pandas as pd

df = pd.read_csv('cleaned_accidents_data.csv')
weather_analysis = df.groupby('weather_conditions')['accident_severity'].value_counts()
print(weather_analysis)
```

### Accidentes Críticos por Edad
```python
critical_accidents = df[df['accident_severity'] == 'Critical']
age_distribution = critical_accidents['driver_age'].describe()
print(age_distribution)
```

## Métricas de Calidad

- **Completitud:** 100% (sin valores faltantes)
- **Consistencia:** 100% (formatos estandarizados)
- **Precisión:** 93.3% (1 registro eliminado por validación)
- **Validez:** 100% (todos los valores en rangos esperados)

## Contribuciones

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar mejoras en el script de limpieza
4. Actualizar documentación
5. Crear Pull Request

## Notas Técnicas

- **Encoding:** UTF-8
- **Separador:** Coma (,)
- **Formato de fechas:** ISO 8601 (YYYY-MM-DD)
- **Precisión decimal:** 1 decimal para edades

## Análisis No Supervisado - Machine Learning

El módulo de análisis no supervisado implementa técnicas de clustering y reducción de dimensionalidad para descubrir patrones ocultos en los datos de accidentes.

### Estructura del Módulo

```
unsupervised_learning/
├── unsupervised_learning_main.py      # Script principal
├── models/
│   ├── clustering_models.py            # Modelos de clustering (K-Means, DBSCAN, Jerárquico)
│   └── dimensionality_reduction.py    # Reducción de dimensionalidad (PCA, t-SNE)
├── preprocessing/
│   └── data_preparation.py            # Preparación de datos para clustering
├── evaluation/
│   ├── metrics_calculation.py         # Cálculo de métricas no supervisadas
│   └── visualizations.py             # Visualizaciones de clusters y componentes
├── optimization/
│   └── hyperparameter_tuning.py      # Optimización de hiperparámetros
└── results/
    ├── clustering_results/            # Resultados de clustering
    └── dimensionality_reduction_results/  # Resultados de reducción de dimensionalidad
```

### Algoritmos Implementados

#### 1. Clustering

**K-Means:**
- Agrupación basada en centroides
- Determinación automática de k óptimo mediante método del codo y silhouette score
- Métricas: Silhouette Score, Calinski-Harabasz, Davies-Bouldin

**DBSCAN:**
- Clustering basado en densidad
- Detección automática de outliers
- Parámetros optimizados: eps y min_samples

**Clustering Jerárquico (Agglomerative):**
- Agrupación jerárquica aglomerativa
- Métodos de linkage: ward, complete, average
- Visualización mediante dendrogramas

#### 2. Reducción de Dimensionalidad

**PCA (Principal Component Analysis):**
- Reducción de dimensionalidad lineal
- Retención del 95% de varianza
- Análisis de componentes principales

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Reducción de dimensionalidad no lineal
- Visualización en 2D/3D
- Preservación de estructura local

### Métricas de Evaluación

- **Silhouette Score:** Mide la calidad de la separación entre clusters (-1 a 1, mayor es mejor)
- **Calinski-Harabasz Index:** Ratio de varianza entre clusters y dentro de clusters (mayor es mejor)
- **Davies-Bouldin Index:** Mide la similitud promedio entre clusters (menor es mejor)
- **Inertia:** Suma de distancias al cuadrado de muestras a su centroide más cercano (solo K-Means)

### Ejecución del Análisis No Supervisado

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar desde la raíz del proyecto:**
   ```bash
   python unsupervised_learning.py
   ```

   O desde dentro de la carpeta:
   ```bash
   cd unsupervised_learning
   python unsupervised_learning_main.py
   ```

### Proceso de Análisis

1. **Preparación de Datos:**
   - Carga y merge de datasets
   - Feature engineering (variables derivadas)
   - Encoding de variables categóricas (One-Hot Encoding)
   - Normalización de features numéricas (StandardScaler)

2. **Reducción de Dimensionalidad:**
   - Aplicación de PCA (12 componentes para 95% varianza)
   - Aplicación de t-SNE para visualización (opcional)

3. **Determinación de K Óptimo:**
   - Método del codo (Elbow Method)
   - Análisis de Silhouette Score
   - Rango de k: 2-10

4. **Clustering:**
   - Entrenamiento de K-Means con k óptimo
   - Entrenamiento de DBSCAN con parámetros optimizados
   - Entrenamiento de Clustering Jerárquico

5. **Evaluación:**
   - Cálculo de métricas para cada modelo
   - Comparación de modelos
   - Identificación del mejor modelo

6. **Visualización:**
   - Gráficos de clusters en 2D/3D (PCA)
   - Visualizaciones t-SNE
   - Dendrogramas (clustering jerárquico)
   - Heatmaps de características de clusters
   - Gráficos de tamaños de clusters

7. **Caracterización:**
   - Análisis de características por cluster
   - Identificación de patrones
   - Exportación de resultados

### Resultados Generados

**Clustering:**
- `metrics_comparison.csv` - Tabla comparativa de métricas
- `metrics_comparison.png` - Gráfico comparativo de métricas
- `kmeans_clusters.png` - Visualización de clusters K-Means
- `kmeans_cluster_sizes.png` - Distribución de tamaños de clusters
- `dbscan_clusters.png` - Visualización de clusters DBSCAN
- `dbscan_cluster_sizes.png` - Distribución de tamaños de clusters
- `hierarchical_clusters.png` - Visualización de clusters jerárquicos
- `hierarchical_dendrogram.png` - Dendrograma jerárquico
- `elbow_method.png` - Método del codo para determinar k
- `silhouette_scores.png` - Scores de silhouette por k
- `cluster_characteristics.csv` - Características de cada cluster
- `cluster_heatmap.png` - Heatmap de características
- `cluster_characteristics_bars.png` - Gráfico de barras de características

**Reducción de Dimensionalidad:**
- `pca_variance_explained.png` - Varianza explicada por componentes
- `pca_components.png` - Visualización de componentes principales
- `component_analysis.csv` - Análisis de componentes principales
- `tsne_visualization.png` - Visualización t-SNE

### Resultados Típicos

Basado en el análisis del dataset de accidentes:

- **Mejor modelo:** DBSCAN (Silhouette Score: ~0.35)
- **K óptimo para K-Means:** 9 clústeres
- **Componentes PCA:** 12 componentes explican 95% de varianza
- **Clusters DBSCAN:** ~47 clusters con ~39% outliers

### Optimización de Hiperparámetros (Opcional)

El módulo incluye optimización automática de hiperparámetros:
- K-Means: optimización de k, init, n_init, max_iter
- DBSCAN: optimización de eps y min_samples
- Clustering Jerárquico: optimización de linkage y metric

**Última actualización:** 2024  
**Versión del dataset:** 1.0  
**Licencia:** Uso académico y de investigación (si hice un commit)
