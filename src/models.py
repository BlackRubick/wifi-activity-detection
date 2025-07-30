"""
Módulo de modelos de machine learning para detección de actividades
==================================================================

Este módulo implementa diferentes modelos para reconocimiento de actividades:
- Modelos tradicionales (SVM, Random Forest, XGBoost)
- Redes neuronales profundas (CNN 1D, CNN 2D, LSTM)
- Modelos híbridos (CNN + LSTM)
- Optimización de hiperparámetros
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, LSTM, GRU, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class ActivityRecognitionModels:
    """
    Clase para entrenar y evaluar modelos de reconocimiento de actividades
    """

    def __init__(self, random_state=42):
        """
        Inicializa el sistema de modelos

        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.history = {}
        self.results = {}

        # Configurar seeds para reproducibilidad
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def prepare_data(self, features, labels, test_size=0.3, validation_size=0.2):
        """
        Prepara los datos para entrenamiento

        Args:
            features: Array de características
            labels: Etiquetas de actividad
            test_size (float): Proporción para test
            validation_size (float): Proporción para validación

        Returns:
            tuple: Datos normalizados para entrenamiento
        """
        print("Preparando datos para entrenamiento...")

        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        self.n_classes = len(np.unique(labels_encoded))
        self.class_names = self.label_encoder.classes_

        # División inicial train/test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            features, labels_encoded, test_size=test_size,
            random_state=self.random_state, stratify=labels_encoded
        )

        # División train/validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size,
            random_state=self.random_state, stratify=y_temp
        )

        # Normalizar características para modelos tradicionales
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Preparar etiquetas categóricas para deep learning
        self.y_train_cat = to_categorical(self.y_train, num_classes=self.n_classes)
        self.y_val_cat = to_categorical(self.y_val, num_classes=self.n_classes)
        self.y_test_cat = to_categorical(self.y_test, num_classes=self.n_classes)

        print(f"Datos preparados:")
        print(f"  Train: {self.X_train.shape[0]} muestras")
        print(f"  Validation: {self.X_val.shape[0]} muestras")
        print(f"  Test: {self.X_test.shape[0]} muestras")
        print(f"  Características: {self.X_train.shape[1]}")
        print(f"  Clases: {self.n_classes}")

        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled

    def train_random_forest(self, n_estimators=100, max_depth=None, **kwargs):
        """
        Entrena modelo Random Forest

        Args:
            n_estimators (int): Número de árboles
            max_depth (int): Profundidad máxima
            **kwargs: Parámetros adicionales

        Returns:
            RandomForestClassifier: Modelo entrenado
        """
        print("Entrenando Random Forest...")

        rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            **kwargs
        )

        # Entrenar
        rf_model.fit(self.X_train_scaled, self.y_train)

        # Evaluar
        train_acc = rf_model.score(self.X_train_scaled, self.y_train)
        val_acc = rf_model.score(self.X_val_scaled, self.y_val)
        test_acc = rf_model.score(self.X_test_scaled, self.y_test)

        # Predicciones
        y_pred_train = rf_model.predict(self.X_train_scaled)
        y_pred_val = rf_model.predict(self.X_val_scaled)
        y_pred_test = rf_model.predict(self.X_test_scaled)

        self.models['random_forest'] = rf_model
        self.results['random_forest'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions_test': y_pred_test,
            'feature_importance': rf_model.feature_importances_
        }

        print(f"Random Forest - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return rf_model

    def train_svm(self, C=1.0, kernel='rbf', **kwargs):
        """
        Entrena modelo SVM

        Args:
            C (float): Parámetro de regularización
            kernel (str): Tipo de kernel
            **kwargs: Parámetros adicionales

        Returns:
            SVC: Modelo entrenado
        """
        print("Entrenando SVM...")

        svm_model = SVC(
            C=C,
            kernel=kernel,
            random_state=self.random_state,
            probability=True,  # Para obtener probabilidades
            **kwargs
        )

        # Entrenar
        svm_model.fit(self.X_train_scaled, self.y_train)

        # Evaluar
        train_acc = svm_model.score(self.X_train_scaled, self.y_train)
        val_acc = svm_model.score(self.X_val_scaled, self.y_val)
        test_acc = svm_model.score(self.X_test_scaled, self.y_test)

        # Predicciones
        y_pred_test = svm_model.predict(self.X_test_scaled)
        y_pred_proba = svm_model.predict_proba(self.X_test_scaled)

        self.models['svm'] = svm_model
        self.results['svm'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions_test': y_pred_test,
            'predictions_proba': y_pred_proba
        }

        print(f"SVM - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return svm_model

    def train_xgboost(self, n_estimators=100, max_depth=6, learning_rate=0.1, **kwargs):
        """
        Entrena modelo XGBoost

        Args:
            n_estimators (int): Número de estimadores
            max_depth (int): Profundidad máxima
            learning_rate (float): Tasa de aprendizaje
            **kwargs: Parámetros adicionales

        Returns:
            XGBClassifier: Modelo entrenado
        """
        print("Entrenando XGBoost...")

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=self.random_state,
            eval_metric='mlogloss',
            **kwargs
        )

        # Entrenar
        xgb_model.fit(self.X_train_scaled, self.y_train)

        # Evaluar
        train_acc = xgb_model.score(self.X_train_scaled, self.y_train)
        val_acc = xgb_model.score(self.X_val_scaled, self.y_val)
        test_acc = xgb_model.score(self.X_test_scaled, self.y_test)

        # Predicciones
        y_pred_test = xgb_model.predict(self.X_test_scaled)
        y_pred_proba = xgb_model.predict_proba(self.X_test_scaled)

        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'predictions_test': y_pred_test,
            'predictions_proba': y_pred_proba,
            'feature_importance': xgb_model.feature_importances_
        }

        print(f"XGBoost - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return xgb_model

    def build_mlp_model(self, hidden_layers=[512, 256, 128], dropout_rate=0.3, activation='relu'):
        """
        Construye modelo MLP (Multi-Layer Perceptron)

        Args:
            hidden_layers (list): Tamaños de capas ocultas
            dropout_rate (float): Tasa de dropout
            activation (str): Función de activación

        Returns:
            Sequential: Modelo MLP
        """
        model = Sequential()

        # Capa de entrada
        model.add(Dense(hidden_layers[0], activation=activation, input_shape=(self.X_train.shape[1],)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Capas ocultas
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Capa de salida
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def build_cnn1d_model(self, input_shape, filters=[64, 128, 256], kernel_size=3, pool_size=2):
        """
        Construye modelo CNN 1D para secuencias temporales

        Args:
            input_shape (tuple): Forma de entrada
            filters (list): Filtros por capa
            kernel_size (int): Tamaño del kernel
            pool_size (int): Tamaño del pooling

        Returns:
            Sequential: Modelo CNN 1D
        """
        model = Sequential()

        # Primera capa convolucional
        model.add(Conv1D(filters[0], kernel_size, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size))

        # Capas convolucionales adicionales
        for f in filters[1:]:
            model.add(Conv1D(f, kernel_size, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size))

        # Capas densas
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def build_lstm_model(self, input_shape, lstm_units=[128, 64], dropout_rate=0.3):
        """
        Construye modelo LSTM para secuencias temporales

        Args:
            input_shape (tuple): Forma de entrada
            lstm_units (list): Unidades LSTM por capa
            dropout_rate (float): Tasa de dropout

        Returns:
            Sequential: Modelo LSTM
        """
        model = Sequential()

        # Primera capa LSTM
        model.add(LSTM(lstm_units[0], return_sequences=len(lstm_units) > 1,
                       input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        # Capas LSTM adicionales
        for i, units in enumerate(lstm_units[1:], 1):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Capas densas
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def build_hybrid_cnn_lstm_model(self, input_shape, cnn_filters=[64, 128],
                                    lstm_units=[128, 64], dropout_rate=0.3):
        """
        Construye modelo híbrido CNN + LSTM

        Args:
            input_shape (tuple): Forma de entrada
            cnn_filters (list): Filtros CNN
            lstm_units (list): Unidades LSTM
            dropout_rate (float): Tasa de dropout

        Returns:
            Sequential: Modelo híbrido
        """
        model = Sequential()

        # Capas CNN
        model.add(Conv1D(cnn_filters[0], 3, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))

        for f in cnn_filters[1:]:
            model.add(Conv1D(f, 3, activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(2))

        # Capas LSTM
        for i, units in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        # Capas de salida
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.n_classes, activation='softmax'))

        return model

    def train_deep_model(self, model, model_name, epochs=100, batch_size=32,
                         learning_rate=0.001, patience=10, verbose=1):
        """
        Entrena un modelo de deep learning

        Args:
            model: Modelo de Keras
            model_name (str): Nombre del modelo
            epochs (int): Número de épocas
            batch_size (int): Tamaño del batch
            learning_rate (float): Tasa de aprendizaje
            patience (int): Paciencia para early stopping
            verbose (int): Nivel de verbosidad

        Returns:
            tuple: (modelo_entrenado, historial)
        """
        print(f"Entrenando {model_name}...")

        # Compilar modelo
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]

        # Entrenar
        history = model.fit(
            self.X_train_scaled, self.y_train_cat,
            validation_data=(self.X_val_scaled, self.y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        # Evaluar
        train_loss, train_acc = model.evaluate(self.X_train_scaled, self.y_train_cat, verbose=0)
        val_loss, val_acc = model.evaluate(self.X_val_scaled, self.y_val_cat, verbose=0)
        test_loss, test_acc = model.evaluate(self.X_test_scaled, self.y_test_cat, verbose=0)

        # Predicciones
        y_pred_test_proba = model.predict(self.X_test_scaled, verbose=0)
        y_pred_test = np.argmax(y_pred_test_proba, axis=1)

        # Guardar resultados
        self.models[model_name] = model
        self.history[model_name] = history
        self.results[model_name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'predictions_test': y_pred_test,
            'predictions_proba': y_pred_test_proba
        }

        print(f"{model_name} - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return model, history

    def hyperparameter_tuning_rf(self, param_grid=None, cv=5):
        """
        Optimización de hiperparámetros para Random Forest

        Args:
            param_grid (dict): Grilla de parámetros
            cv (int): Folds para validación cruzada

        Returns:
            tuple: (mejor_modelo, mejores_parámetros)
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        print("Optimizando hiperparámetros Random Forest...")

        rf = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='accuracy',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)

        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor score CV: {grid_search.best_score_:.4f}")

        # Entrenar modelo con mejores parámetros
        best_rf = grid_search.best_estimator_
        test_acc = best_rf.score(self.X_test_scaled, self.y_test)
        y_pred_test = best_rf.predict(self.X_test_scaled)

        self.models['random_forest_tuned'] = best_rf
        self.results['random_forest_tuned'] = {
            'test_accuracy': test_acc,
            'predictions_test': y_pred_test,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'feature_importance': best_rf.feature_importances_
        }

        return best_rf, grid_search.best_params_

    def evaluate_all_models(self):
        """
        Evalúa todos los modelos entrenados

        Returns:
            pd.DataFrame: Resumen de resultados
        """
        print("\n" + "=" * 60)
        print("EVALUACIÓN DE TODOS LOS MODELOS")
        print("=" * 60)

        results_summary = []

        for model_name, result in self.results.items():
            if 'test_accuracy' in result:
                results_summary.append({
                    'Model': model_name,
                    'Test_Accuracy': result['test_accuracy'],
                    'Train_Accuracy': result.get('train_accuracy', 'N/A'),
                    'Val_Accuracy': result.get('val_accuracy', 'N/A')
                })

        # Crear DataFrame y ordenar por accuracy
        df_results = pd.DataFrame(results_summary)
        df_results = df_results.sort_values('Test_Accuracy', ascending=False)

        print(df_results.to_string(index=False, float_format='%.4f'))

        # Encontrar mejor modelo
        if len(df_results) > 0:
            best_model_name = df_results.iloc[0]['Model']
            best_accuracy = df_results.iloc[0]['Test_Accuracy']
            print(f"\nMejor modelo: {best_model_name} (Accuracy: {best_accuracy:.4f})")

        return df_results

    def plot_training_history(self, model_names=None, figsize=(15, 10)):
        """
        Visualiza el historial de entrenamiento de modelos deep learning

        Args:
            model_names (list): Nombres de modelos a visualizar
            figsize (tuple): Tamaño de la figura
        """
        if model_names is None:
            model_names = list(self.history.keys())

        n_models = len(model_names)
        if n_models == 0:
            print("No hay historial de entrenamiento disponible")
            return

        fig, axes = plt.subplots(n_models, 2, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(1, -1)

        for i, model_name in enumerate(model_names):
            if model_name not in self.history:
                continue

            history = self.history[model_name].history

            # Accuracy
            axes[i, 0].plot(history['accuracy'], label='Train Accuracy', color='blue')
            axes[i, 0].plot(history['val_accuracy'], label='Val Accuracy', color='red')
            axes[i, 0].set_title(f'{model_name} - Accuracy')
            axes[i, 0].set_xlabel('Epoch')
            axes[i, 0].set_ylabel('Accuracy')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)

            # Loss
            axes[i, 1].plot(history['loss'], label='Train Loss', color='blue')
            axes[i, 1].plot(history['val_loss'], label='Val Loss', color='red')
            axes[i, 1].set_title(f'{model_name} - Loss')
            axes[i, 1].set_xlabel('Epoch')
            axes[i, 1].set_ylabel('Loss')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, model_names=None, figsize=(15, 10)):
        """
        Visualiza matrices de confusión para todos los modelos

        Args:
            model_names (list): Nombres de modelos a visualizar
            figsize (tuple): Tamaño de la figura
        """
        if model_names is None:
            model_names = [name for name in self.results.keys()
                           if 'predictions_test' in self.results[name]]

        n_models = len(model_names)
        if n_models == 0:
            print("No hay predicciones disponibles")
            return

        # Calcular número de filas y columnas
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            y_pred = self.results[model_name]['predictions_test']

            # Calcular matriz de confusión
            cm = confusion_matrix(self.y_test, y_pred)

            # Visualizar
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=self.class_names, yticklabels=self.class_names)
            axes[i].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["test_accuracy"]:.4f}')
            axes[i].set_xlabel('Predicción')
            axes[i].set_ylabel('Valor Real')

        # Ocultar axes vacíos
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def generate_classification_reports(self, model_names=None):
        """
        Genera reportes de clasificación detallados

        Args:
            model_names (list): Nombres de modelos a evaluar
        """
        if model_names is None:
            model_names = [name for name in self.results.keys()
                           if 'predictions_test' in self.results[name]]

        for model_name in model_names:
            print(f"\n{'=' * 60}")
            print(f"REPORTE DE CLASIFICACIÓN - {model_name.upper()}")
            print(f"{'=' * 60}")

            y_pred = self.results[model_name]['predictions_test']

            # Reporte detallado
            report = classification_report(
                self.y_test, y_pred,
                target_names=self.class_names,
                output_dict=True
            )

            # Mostrar métricas por clase
            print(f"{'Clase':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 60)

            for class_name in self.class_names:
                if class_name in report:
                    metrics = report[class_name]
                    print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                          f"{metrics['f1-score']:<10.4f} {int(metrics['support']):<10}")

            # Métricas globales
            print("-" * 60)
            print(
                f"{'Accuracy':<15} {'':<10} {'':<10} {report['accuracy']:<10.4f} {int(report['macro avg']['support']):<10}")
            print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<10.4f} {report['macro avg']['recall']:<10.4f} "
                  f"{report['macro avg']['f1-score']:<10.4f} {int(report['macro avg']['support']):<10}")
            print(
                f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<10.4f} {report['weighted avg']['recall']:<10.4f} "
                f"{report['weighted avg']['f1-score']:<10.4f} {int(report['weighted avg']['support']):<10}")


def compare_model_performance(results_dict, figsize=(12, 6)):
    """
    Compara el rendimiento de diferentes modelos

    Args:
        results_dict (dict): Diccionario con resultados de modelos
        figsize (tuple): Tamaño de la figura

    Returns:
        pd.DataFrame: DataFrame con comparación
    """
    metrics_data = []

    for model_name, results in results_dict.items():
        if 'test_accuracy' in results:
            metrics_data.append({
                'Model': model_name,
                'Accuracy': results['test_accuracy'],
                'Type': 'Deep Learning' if model_name in ['mlp', 'cnn1d', 'lstm', 'hybrid'] else 'Traditional'
            })

    df = pd.DataFrame(metrics_data)

    if len(df) == 0:
        print("No hay datos para comparar")
        return df

    # Visualización
    plt.figure(figsize=figsize)

    # Gráfico de barras
    plt.subplot(1, 2, 1)
    colors = ['skyblue' if t == 'Traditional' else 'orange' for t in df['Type']]
    bars = plt.bar(range(len(df)), df['Accuracy'], color=colors)
    plt.xticks(range(len(df)), df['Model'], rotation=45, ha='right')
    plt.ylabel('Test Accuracy')
    plt.title('Comparación de Accuracy por Modelo')
    plt.grid(True, alpha=0.3)

    # Añadir valores en las barras
    for bar, acc in zip(bars, df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Boxplot por tipo
    plt.subplot(1, 2, 2)
    traditional_acc = df[df['Type'] == 'Traditional']['Accuracy'].values
    dl_acc = df[df['Type'] == 'Deep Learning']['Accuracy'].values

    data_to_plot = []
    labels = []
    if len(traditional_acc) > 0:
        data_to_plot.append(traditional_acc)
        labels.append('Traditional')
    if len(dl_acc) > 0:
        data_to_plot.append(dl_acc)
        labels.append('Deep Learning')

    if data_to_plot:
        plt.boxplot(data_to_plot, labels=labels)
        plt.ylabel('Test Accuracy')
        plt.title('Distribución de Accuracy por Tipo')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return df


def analyze_feature_importance(models_dict, feature_names=None, figsize=(15, 10)):
    """
    Analiza la importancia de características en modelos que la soportan

    Args:
        models_dict (dict): Diccionario con resultados de modelos
        feature_names (list): Nombres de las características
        figsize (tuple): Tamaño de la figura
    """
    models_with_importance = ['random_forest', 'xgboost', 'random_forest_tuned']

    # Filtrar modelos que tienen importancia de características
    available_models = [m for m in models_with_importance
                        if m in models_dict and 'feature_importance' in models_dict[m]]

    if not available_models:
        print("No hay modelos con importancia de características disponible")
        return

    plt.figure(figsize=figsize)

    plot_idx = 1
    for model_name in available_models:
        importance = models_dict[model_name]['feature_importance']

        # Obtener índices de las características más importantes
        top_indices = np.argsort(importance)[-20:]  # Top 20
        top_importance = importance[top_indices]

        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f'Feature_{i}' for i in top_indices]

        plt.subplot(len(available_models), 1, plot_idx)
        plt.barh(range(len(top_importance)), top_importance, color='skyblue')
        plt.yticks(range(len(top_importance)), [name[:50] + '...' if len(name) > 50 else name
                                                for name in top_names])
        plt.xlabel('Importancia')
        plt.title(f'Top 20 Características Más Importantes - {model_name}')
        plt.grid(True, alpha=0.3)

        plot_idx += 1

    plt.tight_layout()
    plt.show()


def cross_validation_analysis(X, y, models_dict=None, cv_folds=5, random_state=42):
    """
    Realiza análisis de validación cruzada

    Args:
        X: Características
        y: Etiquetas
        models_dict (dict): Diccionario de modelos (opcional)
        cv_folds (int): Número de folds
        random_state (int): Semilla aleatoria

    Returns:
        dict: Resultados de validación cruzada
    """
    print("Realizando análisis de validación cruzada...")

    cv_results = {}

    # Solo modelos tradicionales (sklearn)
    sklearn_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'SVM': SVC(C=1.0, random_state=random_state),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=random_state, eval_metric='mlogloss')
    }

    for model_name, model in sklearn_models.items():
        print(f"Evaluando {model_name}...")
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

        cv_results[model_name] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }

        print(f"  {model_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # Visualización
    plt.figure(figsize=(10, 6))

    models = list(cv_results.keys())
    means = [cv_results[m]['mean'] for m in models]
    stds = [cv_results[m]['std'] for m in models]

    plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title(f'Validación Cruzada ({cv_folds}-fold)')
    plt.grid(True, alpha=0.3)

    # Añadir valores
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    return cv_results


def save_model_results(model_pipeline, filepath):
    """
    Guarda resultados del pipeline de modelado

    Args:
        model_pipeline: Pipeline de modelos
        filepath (str): Ruta del archivo
    """
    import pickle

    # Preparar datos para guardado (sin modelos de Keras que son grandes)
    data_to_save = {
        'results': model_pipeline.results,
        'label_encoder': model_pipeline.label_encoder,
        'scaler': model_pipeline.scaler,
        'n_classes': model_pipeline.n_classes,
        'class_names': model_pipeline.class_names,
        'random_state': model_pipeline.random_state
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Resultados guardados en: {filepath}")


def create_model_comparison_report(model_pipeline):
    """
    Crea un reporte comparativo detallado de todos los modelos

    Args:
        model_pipeline: Pipeline de modelos
    """
    print("\n" + "=" * 80)
    print("REPORTE COMPARATIVO DE MODELOS")
    print("=" * 80)

    # Tabla resumen
    results_data = []
    for model_name, results in model_pipeline.results.items():
        if 'test_accuracy' in results:
            results_data.append({
                'Modelo': model_name,
                'Accuracy_Test': results['test_accuracy'],
                'Accuracy_Train': results.get('train_accuracy', 'N/A'),
                'Accuracy_Val': results.get('val_accuracy', 'N/A'),
                'Tipo': 'Deep Learning' if model_name in ['mlp', 'cnn1d', 'lstm', 'hybrid'] else 'Tradicional'
            })

    if not results_data:
        print("No hay resultados para mostrar")
        return

    df_results = pd.DataFrame(results_data)
    df_results = df_results.sort_values('Accuracy_Test', ascending=False)

    print("\nTABLA COMPARATIVA:")
    print(df_results.to_string(index=False, float_format='%.4f'))

    # Análisis por tipo
    print(f"\nANÁLISIS POR TIPO DE MODELO:")
    for tipo in df_results['Tipo'].unique():
        subset = df_results[df_results['Tipo'] == tipo]
        print(f"\n{tipo}:")
        print(f"  Mejor accuracy: {subset['Accuracy_Test'].max():.4f}")
        print(f"  Promedio: {subset['Accuracy_Test'].mean():.4f}")
        print(f"  Desv. estándar: {subset['Accuracy_Test'].std():.4f}")

    # Recomendaciones
    print(f"\nRECOMENDACIONES:")
    best_model = df_results.iloc[0]
    print(f"- Mejor modelo general: {best_model['Modelo']} ({best_model['Accuracy_Test']:.4f})")

    if best_model['Tipo'] == 'Tradicional':
        print("- Los modelos tradicionales funcionan bien para este problema")
        print("- Considera usar el modelo con mejor interpretabilidad")
    else:
        print("- Los modelos de deep learning superan a los tradicionales")
        print("- Considera el costo computacional vs. mejora en rendimiento")

    # Verificar overfitting
    for _, row in df_results.iterrows():
        if row['Accuracy_Train'] != 'N/A':
            train_acc = float(row['Accuracy_Train'])
            test_acc = float(row['Accuracy_Test'])
            if train_acc - test_acc > 0.1:
                print(f"- ADVERTENCIA: {row['Modelo']} muestra signos de overfitting")


def demo_complete_modeling():
    """
    Demostración completa del pipeline de modelado
    """
    print("=== DEMOSTRACIÓN COMPLETA DE MODELADO ===")

    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    n_classes = 4

    # Crear características sintéticas con patrones
    X = np.random.randn(n_samples, n_features)

    # Crear etiquetas con patrones específicos
    y = np.zeros(n_samples)
    for i in range(n_samples):
        # Diferentes patrones para diferentes clases
        if np.mean(X[i, :20]) > 0:  # Clase basada en primeras 20 features
            if np.std(X[i, 20:40]) > 1:
                y[i] = 0
            else:
                y[i] = 1
        else:
            if np.mean(X[i, 40:60]) > 0:
                y[i] = 2
            else:
                y[i] = 3

    print(f"Datos sintéticos: {X.shape}")
    print(f"Distribución de clases: {np.bincount(y.astype(int))}")

    # Crear pipeline de modelado
    model_pipeline = ActivityRecognitionModels(random_state=42)

    # Preparar datos
    X_train_scaled, X_val_scaled, X_test_scaled = model_pipeline.prepare_data(X, y)

    # Entrenar modelos tradicionales
    print("\n--- ENTRENANDO MODELOS TRADICIONALES ---")
    model_pipeline.train_random_forest(n_estimators=100, max_depth=10)
    model_pipeline.train_svm(C=1.0, kernel='rbf')
    model_pipeline.train_xgboost(n_estimators=100, max_depth=6)

    # Entrenar modelos de deep learning
    print("\n--- ENTRENANDO MODELOS DEEP LEARNING ---")

    # MLP
    mlp_model = model_pipeline.build_mlp_model(hidden_layers=[256, 128, 64])
    model_pipeline.train_deep_model(mlp_model, 'mlp', epochs=50, batch_size=32, verbose=0)

    # Optimización de hiperparámetros
    print("\n--- OPTIMIZACIÓN DE HIPERPARÁMETROS ---")
    best_rf, best_params = model_pipeline.hyperparameter_tuning_rf()

    # Evaluación completa
    print("\n--- EVALUACIÓN COMPLETA ---")
    results_df = model_pipeline.evaluate_all_models()

    # Visualizaciones
    print("\n--- VISUALIZACIONES ---")
    model_pipeline.plot_training_history()
    model_pipeline.plot_confusion_matrices()
    model_pipeline.generate_classification_reports()

    # Análisis comparativo
    performance_df = compare_model_performance(model_pipeline.results)

    # Análisis de validación cruzada
    cv_results = cross_validation_analysis(X_train_scaled, model_pipeline.y_train)

    print("\n=== RESUMEN FINAL ===")
    if len(results_df) > 0:
        best_model = results_df.iloc[0]
        print(f"Mejor modelo: {best_model['Model']}")
        print(f"Accuracy de prueba: {best_model['Test_Accuracy']:.4f}")

    # Crear reporte final
    create_model_comparison_report(model_pipeline)

    return model_pipeline


if __name__ == "__main__":
    # Ejecutar demostración completa
    model_pipeline = demo_complete_modeling()

    print("\n🎉 Modelado completado exitosamente!")
    print(f"📊 {len(model_pipeline.results)} modelos entrenados")
    if model_pipeline.results:
        best_acc = max(r['test_accuracy'] for r in model_pipeline.results.values() if 'test_accuracy' in r)
        print(f"🏆 Mejor accuracy: {best_acc:.4f}")