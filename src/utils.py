"""
Módulo de utilidades generales para el proyecto
==============================================

Este módulo contiene funciones de utilidad común:
- Manejo de archivos y configuración
- Logging y reportes
- Validación y verificación
- Funciones auxiliares
"""

import os
import pickle
import json
import yaml
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configura el sistema de logging del proyecto

    Args:
        log_level: Nivel de logging
        log_file (str): Archivo de log (opcional)
    """
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configurar handlers
    handlers = [logging.StreamHandler()]

    if log_file:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configurar logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )

    return logging.getLogger(__name__)


def create_project_structure(base_path='.'):
    """
    Crea la estructura de directorios del proyecto

    Args:
        base_path (str): Ruta base del proyecto
    """
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'results/figures',
        'results/models',
        'results/metrics',
        'results/evaluation',
        'logs',
        'configs'
    ]

    for directory in directories:
        path = Path(base_path) / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado: {path}")


def save_config(config_dict, filepath):
    """
    Guarda configuración en archivo JSON o YAML

    Args:
        config_dict (dict): Diccionario de configuración
        filepath (str): Ruta del archivo
    """
    filepath = Path(filepath)

    # Crear directorio si no existe
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    print(f"Configuración guardada en: {filepath}")


def load_config(filepath):
    """
    Carga configuración desde archivo JSON o YAML

    Args:
        filepath (str): Ruta del archivo

    Returns:
        dict: Diccionario de configuración
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {filepath}")

    if filepath.suffix.lower() in ['.yaml', '.yml']:
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            config = json.load(f)

    print(f"Configuración cargada desde: {filepath}")
    return config


def save_pickle(obj, filepath, create_dirs=True):
    """
    Guarda objeto usando pickle

    Args:
        obj: Objeto a guardar
        filepath (str): Ruta del archivo
        create_dirs (bool): Crear directorios si no existen
    """
    filepath = Path(filepath)

    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

    print(f"Objeto guardado en: {filepath}")


def load_pickle(filepath):
    """
    Carga objeto usando pickle

    Args:
        filepath (str): Ruta del archivo

    Returns:
        object: Objeto cargado
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    with open(filepath, 'rb') as f:
        obj = pickle.load(f)

    print(f"Objeto cargado desde: {filepath}")
    return obj


def validate_data_shape(data, expected_dims, name="data"):
    """
    Valida las dimensiones de los datos

    Args:
        data: Array de datos
        expected_dims (int): Número esperado de dimensiones
        name (str): Nombre del conjunto de datos

    Raises:
        ValueError: Si las dimensiones no coinciden
    """
    if len(data.shape) != expected_dims:
        raise ValueError(f"{name} debe tener {expected_dims} dimensiones, "
                        f"pero tiene {len(data.shape)}: {data.shape}")

    print(f"✓ {name} tiene forma válida: {data.shape}")


def validate_labels_consistency(labels, data, name="labels"):
    """
    Valida consistencia entre etiquetas y datos

    Args:
        labels: Array de etiquetas
        data: Array de datos
        name (str): Nombre de las etiquetas

    Raises:
        ValueError: Si hay inconsistencias
    """
    # Verificar que no hay NaN
    if np.any(np.isnan(labels)):
        raise ValueError(f"{name} contiene valores NaN")

    # Verificar rango de etiquetas
    unique_labels = np.unique(labels)
    if np.min(unique_labels) < 0:
        raise ValueError(f"{name} contiene valores negativos")

    print(f"✓ {name} son válidas: {len(unique_labels)} clases únicas")


def check_data_quality(data, name="data", max_nan_percentage=5.0):
    """
    Verifica la calidad de los datos

    Args:
        data: Array de datos
        name (str): Nombre del conjunto de datos
        max_nan_percentage (float): Porcentaje máximo de NaN permitido

    Returns:
        dict: Reporte de calidad
    """
    quality_report = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'total_elements': data.size,
        'nan_count': np.sum(np.isnan(data)),
        'inf_count': np.sum(np.isinf(data)),
        'min_value': np.nanmin(data),
        'max_value': np.nanmax(data),
        'mean_value': np.nanmean(data),
        'std_value': np.nanstd(data)
    }

    # Calcular porcentajes
    quality_report['nan_percentage'] = (quality_report['nan_count'] / quality_report['total_elements']) * 100
    quality_report['inf_percentage'] = (quality_report['inf_count'] / quality_report['total_elements']) * 100

    print(f"=== REPORTE DE CALIDAD: {name} ===")
    print(f"Forma: {quality_report['shape']}")
    print(f"Tipo: {quality_report['dtype']}")
    print(f"Elementos totales: {quality_report['total_elements']:,}")
    print(f"Valores NaN: {quality_report['nan_count']:,} ({quality_report['nan_percentage']:.2f}%)")
    print(f"Valores infinitos: {quality_report['inf_count']:,} ({quality_report['inf_percentage']:.2f}%)")
    print(f"Rango: [{quality_report['min_value']:.6f}, {quality_report['max_value']:.6f}]")
    print(f"Media: {quality_report['mean_value']:.6f}")
    print(f"Desv. estándar: {quality_report['std_value']:.6f}")

    # Advertencias
    if quality_report['nan_percentage'] > max_nan_percentage:
        print(f"⚠️ ADVERTENCIA: {quality_report['nan_percentage']:.2f}% de valores NaN (>{max_nan_percentage}%)")

    if quality_report['inf_count'] > 0:
        print(f"⚠️ ADVERTENCIA: {quality_report['inf_count']} valores infinitos detectados")

    if quality_report['std_value'] == 0:
        print(f"⚠️ ADVERTENCIA: Desviación estándar es 0 (datos constantes)")

    return quality_report


def memory_usage_report(objects_dict):
    """
    Genera reporte de uso de memoria

    Args:
        objects_dict (dict): Diccionario de objetos a analizar

    Returns:
        pd.DataFrame: Reporte de memoria
    """
    memory_data = []

    for name, obj in objects_dict.items():
        if hasattr(obj, 'nbytes'):
            # Para arrays numpy
            memory_mb = obj.nbytes / (1024**2)
            size_info = str(obj.shape)
            dtype_info = str(obj.dtype)
        elif hasattr(obj, 'memory_usage'):
            # Para DataFrames pandas
            memory_mb = obj.memory_usage(deep=True).sum() / (1024**2)
            size_info = str(obj.shape)
            dtype_info = str(obj.dtypes.iloc[0]) if len(obj.dtypes) > 0 else 'mixed'
        else:
            # Para otros objetos
            try:
                import sys
                memory_mb = sys.getsizeof(obj) / (1024**2)
                size_info = str(len(obj)) if hasattr(obj, '__len__') else 'N/A'
                dtype_info = str(type(obj).__name__)
            except:
                memory_mb = 0
                size_info = 'N/A'
                dtype_info = 'unknown'

        memory_data.append({
            'Object': name,
            'Memory (MB)': memory_mb,
            'Size': size_info,
            'Type': dtype_info
        })

    df = pd.DataFrame(memory_data).sort_values('Memory (MB)', ascending=False)

    print("=== REPORTE DE USO DE MEMORIA ===")
    print(df.to_string(index=False, float_format='%.2f'))
    print(f"\nMemoria total: {df['Memory (MB)'].sum():.2f} MB")

    return df


def plot_data_distribution(data, title="Distribución de Datos", bins=50, figsize=(12, 8)):
    """
    Visualiza la distribución de los datos

    Args:
        data: Array de datos
        title (str): Título del gráfico
        bins (int): Número de bins para histograma
        figsize (tuple): Tamaño de la figura
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Aplanar datos si es multidimensional
    if len(data.shape) > 1:
        flat_data = data.flatten()
    else:
        flat_data = data

    # Eliminar NaN e infinitos para visualización
    clean_data = flat_data[np.isfinite(flat_data)]

    # Histograma
    axes[0, 0].hist(clean_data, bins=bins, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histograma')
    axes[0, 0].set_xlabel('Valor')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    axes[0, 1].boxplot(clean_data)
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Valor')
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot (aproximado)
    from scipy import stats
    stats.probplot(clean_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal)')
    axes[1, 0].grid(True, alpha=0.3)

    # Estadísticas
    stats_text = f"""
    Estadísticas:
    Media: {np.mean(clean_data):.4f}
    Mediana: {np.median(clean_data):.4f}
    Std: {np.std(clean_data):.4f}
    Min: {np.min(clean_data):.4f}
    Max: {np.max(clean_data):.4f}
    Skewness: {stats.skew(clean_data):.4f}
    Kurtosis: {stats.kurtosis(clean_data):.4f}
    """

    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    axes[1, 1].set_title('Estadísticas')
    axes[1, 1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def generate_timestamp():
    """
    Genera timestamp formateado

    Returns:
        str: Timestamp en formato YYYY-MM-DD_HH-MM-SS
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_experiment_folder(base_path="experiments", experiment_name=None):
    """
    Crea carpeta para experimento con timestamp

    Args:
        base_path (str): Ruta base para experimentos
        experiment_name (str): Nombre del experimento (opcional)

    Returns:
        Path: Ruta del experimento
    """
    if experiment_name is None:
        experiment_name = f"experiment_{generate_timestamp()}"
    else:
        experiment_name = f"{experiment_name}_{generate_timestamp()}"

    experiment_path = Path(base_path) / experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Crear subdirectorios
    subdirs = ['models', 'figures', 'logs', 'configs']
    for subdir in subdirs:
        (experiment_path / subdir).mkdir(exist_ok=True)

    print(f"Experimento creado en: {experiment_path}")
    return experiment_path


def compare_arrays(arr1, arr2, name1="Array 1", name2="Array 2", tolerance=1e-6):
    """
    Compara dos arrays y reporta diferencias

    Args:
        arr1, arr2: Arrays a comparar
        name1, name2 (str): Nombres de los arrays
        tolerance (float): Tolerancia para comparación

    Returns:
        dict: Reporte de comparación
    """
    comparison = {
        'shapes_equal': arr1.shape == arr2.shape,
        'dtypes_equal': arr1.dtype == arr2.dtype,
        'arrays_equal': False,
        'max_difference': None,
        'mean_difference': None,
        'different_elements': 0
    }

    print(f"=== COMPARACIÓN: {name1} vs {name2} ===")
    print(f"Forma {name1}: {arr1.shape}")
    print(f"Forma {name2}: {arr2.shape}")
    print(f"Formas iguales: {comparison['shapes_equal']}")
    print(f"Tipos iguales: {comparison['dtypes_equal']}")

    if comparison['shapes_equal']:
        # Calcular diferencias
        diff = np.abs(arr1 - arr2)
        comparison['max_difference'] = np.max(diff)
        comparison['mean_difference'] = np.mean(diff)
        comparison['different_elements'] = np.sum(diff > tolerance)
        comparison['arrays_equal'] = comparison['max_difference'] <= tolerance

        print(f"Arrays iguales (tolerancia={tolerance}): {comparison['arrays_equal']}")
        print(f"Diferencia máxima: {comparison['max_difference']:.8f}")
        print(f"Diferencia promedio: {comparison['mean_difference']:.8f}")
        print(f"Elementos diferentes: {comparison['different_elements']}")
    else:
        print("No se puede comparar: formas diferentes")

    return comparison


def set_random_seeds(seed=42):
    """
    Establece semillas aleatorias para reproducibilidad

    Args:
        seed (int): Semilla aleatoria
    """
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    print(f"Semillas aleatorias establecidas: {seed}")


def format_duration(seconds):
    """
    Formatea duración en segundos a formato legible

    Args:
        seconds (float): Duración en segundos

    Returns:
        str: Duración formateada
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def timer_decorator(func):
    """
    Decorador para medir tiempo de ejecución

    Args:
        func: Función a decorar

    Returns:
        function: Función decorada
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        print(f"⏱️ {func.__name__} ejecutado en {format_duration(duration)}")
        return result

    return wrapper


class ProgressBar:
    """
    Barra de progreso simple
    """

    def __init__(self, total, width=50, prefix="Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, increment=1):
        """Actualiza la barra de progreso"""
        self.current += increment
        self._print_bar()

    def _print_bar(self):
        """Imprime la barra de progreso"""
        if self.total == 0:
            percentage = 100
        else:
            percentage = (self.current / self.total) * 100

        filled = int(self.width * self.current // self.total)
        bar = "█" * filled + "░" * (self.width - filled)

        print(f"\r{self.prefix}: |{bar}| {percentage:.1f}% ({self.current}/{self.total})", end="")

        if self.current >= self.total:
            print()  # Nueva línea al completar


class ExperimentTracker:
    """
    Tracker para experimentos de ML
    """

    def __init__(self, experiment_name, base_path="experiments"):
        self.experiment_name = experiment_name
        self.experiment_path = create_experiment_folder(base_path, experiment_name)
        self.results = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'experiment_name': experiment_name
        }

    def log_metric(self, metric_name, value, step=None):
        """Registra una métrica"""
        if metric_name not in self.results:
            self.results[metric_name] = []

        self.results[metric_name].append({
            'value': value,
            'step': step,
            'timestamp': datetime.now().isoformat()
        })

    def log_params(self, params_dict):
        """Registra parámetros del experimento"""
        self.metadata['params'] = params_dict

    def save_artifact(self, obj, filename):
        """Guarda un artefacto del experimento"""
        filepath = self.experiment_path / filename
        save_pickle(obj, filepath)

    def save_figure(self, fig, filename):
        """Guarda una figura"""
        filepath = self.experiment_path / 'figures' / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def finalize(self):
        """Finaliza el experimento y guarda resultados"""
        self.metadata['finished_at'] = datetime.now().isoformat()

        # Guardar resultados y metadata
        save_pickle(self.results, self.experiment_path / 'results.pkl')
        save_config(self.metadata, self.experiment_path / 'metadata.json')

        print(f"Experimento finalizado: {self.experiment_path}")


def validate_file_path(filepath, extensions=None, must_exist=True):
    """
    Valida ruta de archivo

    Args:
        filepath (str): Ruta del archivo
        extensions (list): Extensiones permitidas
        must_exist (bool): Si el archivo debe existir

    Raises:
        ValueError: Si la validación falla
    """
    filepath = Path(filepath)

    if must_exist and not filepath.exists():
        raise ValueError(f"Archivo no encontrado: {filepath}")

    if extensions:
        if filepath.suffix.lower() not in [ext.lower() for ext in extensions]:
            raise ValueError(f"Extensión no válida. Permitidas: {extensions}")

    print(f"✓ Ruta válida: {filepath}")


def clean_data_artifacts(data, remove_nan=True, remove_inf=True, fill_value=0):
    """
    Limpia artefactos comunes en los datos

    Args:
        data: Array de datos
        remove_nan (bool): Eliminar valores NaN
        remove_inf (bool): Eliminar valores infinitos
        fill_value: Valor para reemplazar artefactos

    Returns:
        np.ndarray: Datos limpios
    """
    clean_data = data.copy()

    original_size = data.size
    artifacts_count = 0

    if remove_nan:
        nan_mask = np.isnan(clean_data)
        artifacts_count += np.sum(nan_mask)
        clean_data[nan_mask] = fill_value

    if remove_inf:
        inf_mask = np.isinf(clean_data)
        artifacts_count += np.sum(inf_mask)
        clean_data[inf_mask] = fill_value

    if artifacts_count > 0:
        percentage = (artifacts_count / original_size) * 100
        print(f"Artefactos limpiados: {artifacts_count} ({percentage:.2f}%) -> valor: {fill_value}")

    return clean_data


def create_summary_report(project_info, results_dict, output_path="project_summary.txt"):
    """
    Crea reporte resumen del proyecto

    Args:
        project_info (dict): Información del proyecto
        results_dict (dict): Diccionario de resultados
        output_path (str): Ruta del archivo de salida
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content = f"""
# Reporte Resumen del Proyecto WiFi Activity Detection
# Generado: {timestamp}

## Información del Proyecto
"""

    for key, value in project_info.items():
        report_content += f"{key}: {value}\n"

    report_content += "\n## Resultados\n"

    for key, value in results_dict.items():
        if isinstance(value, dict):
            report_content += f"\n### {key}\n"
            for subkey, subvalue in value.items():
                report_content += f"  {subkey}: {subvalue}\n"
        else:
            report_content += f"{key}: {value}\n"

    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"Reporte guardado en: {output_path}")


# Configuraciones por defecto
DEFAULT_CONFIG = {
    'data': {
        'sampling_rate': 1000,
        'window_size': 1000,
        'overlap': 0.5
    },
    'preprocessing': {
        'bandpass_filter': {
            'lowcut': 1,
            'highcut': 50,
            'order': 4
        },
        'normalization': 'standard',
        'outlier_removal': {
            'method': 'zscore',
            'threshold': 3
        }
    },
    'feature_extraction': {
        'extract_temporal': True,
        'extract_frequency': True,
        'extract_timefrequency': True,
        'extract_correlation': True,
        'extract_wavelet': True
    },
    'modeling': {
        'test_size': 0.3,
        'validation_size': 0.2,
        'random_state': 42
    }
}


def get_default_config():
    """
    Retorna configuración por defecto

    Returns:
        dict: Configuración por defecto
    """
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)


def demo_utilities():
    """
    Demostración de las utilidades
    """
    print("=== DEMOSTRACIÓN DE UTILIDADES ===")

    # Crear estructura de proyecto
    print("\n1. Creando estructura de proyecto...")
    create_project_structure("demo_project")

    # Configurar logging
    print("\n2. Configurando logging...")
    logger = setup_logging(log_file="demo_project/logs/demo.log")
    logger.info("Sistema de logging configurado")

    # Generar datos de prueba
    print("\n3. Generando datos de prueba...")
    np.random.seed(42)
    test_data = np.random.randn(1000, 50)
    test_labels = np.random.randint(0, 3, 1000)

    # Validar datos
    print("\n4. Validando datos...")
    validate_data_shape(test_data, 2, "test_data")
    validate_labels_consistency(test_labels, test_data, "test_labels")

    # Verificar calidad
    print("\n5. Verificando calidad de datos...")
    quality_report = check_data_quality(test_data, "test_data")

    # Visualizar distribución
    print("\n6. Visualizando distribución...")
    plot_data_distribution(test_data, "Datos de Prueba")

    # Reporte de memoria
    print("\n7. Analizando uso de memoria...")
    objects = {
        'test_data': test_data,
        'test_labels': test_labels,
        'quality_report': quality_report
    }
    memory_report = memory_usage_report(objects)

    # Guardar configuración
    print("\n8. Guardando configuración...")
    config = get_default_config()
    save_config(config, "demo_project/configs/default_config.json")

    # Crear experimento
    print("\n9. Creando experimento...")
    experiment_path = create_experiment_folder("demo_project/experiments", "demo_test")

    # Configurar semillas
    print("\n10. Configurando reproducibilidad...")
    set_random_seeds(42)

    # Ejemplo de ExperimentTracker
    print("\n11. Demostrando ExperimentTracker...")
    tracker = ExperimentTracker("demo_experiment")
    tracker.log_params({'learning_rate': 0.01, 'batch_size': 32})
    tracker.log_metric('accuracy', 0.85, step=1)
    tracker.log_metric('loss', 0.3, step=1)
    tracker.finalize()

    print("\n✅ Demostración de utilidades completada!")
    return {
        'test_data': test_data,
        'test_labels': test_labels,
        'quality_report': quality_report,
        'experiment_path': experiment_path,
        'tracker': tracker
    }


if __name__ == "__main__":
    # Ejecutar demostración
    demo_results = demo_utilities()
    print(f"\n🎉 Utilidades demostradas exitosamente!")
    print(f"📊 Datos generados: {demo_results['test_data'].shape}")
    print(f"📁 Experimento en: {demo_results['experiment_path']}")
    print(f"📋 Tracker: {demo_results['tracker'].experiment_name}")