# Dataset de Accidentes Movilísticos

Este repositorio contiene un dataset limpio de accidentes movilísticos generado a partir de datos brutos mediante un proceso automatizado de limpieza y validación de datos. El dataset está diseñado para análisis de extracción de conocimientos y minería de datos.

## 📁 Estructura del Repositorio

```
dataset/
├── raw_accidents_data.csv          # Dataset original (datos brutos)
├── cleaned_accidents_data.csv      # Dataset limpio (procesado)
├── data_cleaning_script.py         # Script de limpieza automatizada
├── requirements.txt                # Dependencias de Python
├── .gitignore                      # Archivos a ignorar en Git
└── README.md                      # Este archivo
```

## Especificaciones del Dataset

### Dataset Limpio (`cleaned_accidents_data.csv`)

**Registros:** 104 accidentes  
**Período:** Enero 2022 - Diciembre 2024  
**Calidad:** 99.0% de datos conservados (1 registro eliminado por edad inválida)

#### Columnas del Dataset

| Columna | Tipo | Descripción | Valores Posibles |
|---------|------|-------------|------------------|
| `incident_id` | Entero | Identificador único del accidente | 101-205 |
| `incident_date` | Fecha | Fecha del accidente | Formato: YYYY-MM-DD |
| `driver_age` | Decimal | Edad del conductor | 18-80 años |
| `road_conditions` | Categórico | Condiciones del camino | Dry, Wet, Icy |
| `weather_conditions` | Categórico | Condiciones climáticas | Sunny, Rain, Snow, Cloudy, Clear |
| `accident_severity` | Categórico | Severidad del accidente | Minor, Serious, Critical |
| `number_of_vehicles` | Entero | Número de vehículos involucrados | 1-5 |
| `number_of_fatalities` | Entero | Número de fatalidades | 0-3 |

#### Estadísticas del Dataset

- **Rango de fechas:** 2022-01-07 a 2024-12-22
- **Edad promedio de conductores:** 48.7 años
- **Total de vehículos involucrados:** 327
- **Total de fatalidades:** 270

#### Distribución por Severidad

- **Serious:** 52 accidentes (50.0%)
- **Minor:** 35 accidentes (33.7%)
- **Critical:** 17 accidentes (16.3%)

## 🛠️ Proceso de Limpieza

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

## 🚀 Cómo Generar el Dataset Limpio

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

## 📊 Casos de Uso para Extracción de Conocimientos

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

## 🔍 Ejemplos de Consultas

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

## 📈 Métricas de Calidad

- **Completitud:** 100% (sin valores faltantes)
- **Consistencia:** 100% (formatos estandarizados)
- **Precisión:** 93.3% (1 registro eliminado por validación)
- **Validez:** 100% (todos los valores en rangos esperados)

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar mejoras en el script de limpieza
4. Actualizar documentación
5. Crear Pull Request

## 📝 Notas Técnicas

- **Encoding:** UTF-8
- **Separador:** Coma (,)
- **Formato de fechas:** ISO 8601 (YYYY-MM-DD)
- **Precisión decimal:** 1 decimal para edades

**Última actualización:** 2024  
**Versión del dataset:** 1.0  
**Licencia:** Uso académico y de investigación
