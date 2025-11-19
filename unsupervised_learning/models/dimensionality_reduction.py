#!/usr/bin/env python3
"""
Reducción de Dimensionalidad
==============================

Este módulo implementa los algoritmos de reducción de dimensionalidad:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

class DimensionalityReduction:
    """
    Clase que encapsula los modelos de reducción de dimensionalidad
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa la clase de reducción de dimensionalidad
        
        Args:
            random_state (int): Semilla aleatoria para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.transformations = {}
        self.variance_explained = {}
        
    def fit_pca(self, X, n_components=None, variance_threshold=0.95):
        """
        Aplica PCA a los datos
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            n_components (int): Número de componentes (None para auto-determinar)
            variance_threshold (float): Umbral de varianza explicada (0.95 = 95%)
            
        Returns:
            PCA: Modelo PCA entrenado
        """
        print("Aplicando PCA...")
        
        # Si n_components no se especifica, determinar automáticamente
        if n_components is None:
            # Primero ajustar con todos los componentes para ver varianza
            pca_temp = PCA()
            pca_temp.fit(X)
            
            # Calcular varianza acumulada
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            
            # Encontrar número de componentes que explican el umbral de varianza
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            
            print(f"  Componentes necesarios para {variance_threshold*100:.1f}% varianza: {n_components}")
        
        # Aplicar PCA con el número de componentes determinado
        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_transformed = pca.fit_transform(X)
        
        # Guardar modelo y transformación
        self.models['PCA'] = pca
        self.transformations['PCA'] = X_transformed
        
        # Calcular varianza explicada
        variance_explained = pca.explained_variance_ratio_
        cumsum_variance = np.cumsum(variance_explained)
        
        self.variance_explained['PCA'] = {
            'individual': variance_explained,
            'cumulative': cumsum_variance,
            'total_variance': cumsum_variance[-1]
        }
        
        print(f"  PCA completado: {n_components} componentes")
        print(f"  Varianza total explicada: {cumsum_variance[-1]*100:.2f}%")
        print(f"  Varianza del PC1: {variance_explained[0]*100:.2f}%")
        print(f"  Varianza del PC2: {variance_explained[1]*100:.2f}%")
        
        return pca
    
    def fit_tsne(self, X, n_components=2, perplexity=30, learning_rate=200, max_iter=1000):
        """
        Aplica t-SNE a los datos (solo para visualización)
        
        Args:
            X (pd.DataFrame o np.array): Features normalizadas
            n_components (int): Número de dimensiones (2 o 3 para visualización)
            perplexity (float): Número de vecinos cercanos (típicamente 5-50)
            learning_rate (float): Tasa de aprendizaje (típicamente 10-1000)
            max_iter (int): Número máximo de iteraciones
            
        Returns:
            TSNE: Modelo t-SNE entrenado
        """
        print(f"Aplicando t-SNE (n_components={n_components}, perplexity={perplexity})...")
        print("  Nota: t-SNE puede tardar varios minutos en datasets grandes...")
        
        # Reducir dimensionalidad primero con PCA si hay muchas features (acelera t-SNE)
        if X.shape[1] > 50:
            print("  Reduciendo dimensionalidad con PCA primero (50 componentes)...")
            pca_pre = PCA(n_components=50, random_state=self.random_state)
            X_pca = pca_pre.fit_transform(X)
            print(f"  Datos reducidos de {X.shape[1]} a 50 dimensiones con PCA")
        else:
            X_pca = X
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=self.random_state,
            verbose=1
        )
        
        X_transformed = tsne.fit_transform(X_pca)
        
        # Guardar modelo y transformación
        self.models['t-SNE'] = tsne
        self.transformations['t-SNE'] = X_transformed
        
        print(f"  t-SNE completado: {n_components} dimensiones")
        
        return tsne
    
    def get_pca_components_info(self, feature_names=None, top_n=10):
        """
        Obtiene información sobre los componentes principales
        
        Args:
            feature_names (list): Nombres de las features originales
            top_n (int): Número de features más importantes a mostrar por componente
            
        Returns:
            dict: Información de componentes principales
        """
        if 'PCA' not in self.models:
            raise ValueError("PCA no ha sido entrenado")
        
        pca = self.models['PCA']
        components = pca.components_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(components.shape[1])]
        
        components_info = {}
        
        for i in range(min(top_n, components.shape[0])):
            # Obtener índices de las features con mayor peso absoluto
            abs_weights = np.abs(components[i])
            top_indices = np.argsort(abs_weights)[-top_n:][::-1]
            
            component_info = {
                'variance_explained': self.variance_explained['PCA']['individual'][i],
                'top_features': [
                    {
                        'feature': feature_names[idx],
                        'weight': components[i][idx]
                    }
                    for idx in top_indices
                ]
            }
            
            components_info[f'PC{i+1}'] = component_info
        
        return components_info
    
    def transform(self, X, method='PCA'):
        """
        Transforma nuevos datos usando el modelo entrenado
        
        Args:
            X (pd.DataFrame o np.array): Features a transformar
            method (str): Método a usar ('PCA' o 't-SNE')
            
        Returns:
            np.array: Datos transformados
        """
        if method not in self.models:
            raise ValueError(f"Método {method} no ha sido entrenado")
        
        if method == 'PCA':
            return self.models['PCA'].transform(X)
        elif method == 't-SNE':
            # t-SNE no puede transformar nuevos datos, solo usar fit_transform
            raise ValueError("t-SNE no puede transformar nuevos datos. Use fit_tsne() directamente.")
        else:
            raise ValueError(f"Método {method} no reconocido")
    
    def get_variance_explained_plot_data(self):
        """
        Obtiene datos para graficar varianza explicada
        
        Returns:
            dict: Datos para visualización
        """
        if 'PCA' not in self.variance_explained:
            raise ValueError("PCA no ha sido entrenado")
        
        return {
            'individual': self.variance_explained['PCA']['individual'],
            'cumulative': self.variance_explained['PCA']['cumulative'],
            'n_components': len(self.variance_explained['PCA']['individual'])
        }

