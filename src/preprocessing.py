"""
Módulo de preprocesamiento de señales CSI
========================================

Este módulo contiene todas las funcionalidades para preprocesar señales CSI:
- Filtrado de ruido y suavizado
- Normalización y estandarización
- Segmentación en ventanas temporales
- Eliminación de artefactos y outliers
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')


class CSIPreprocessor:
    """
    Clase para preprocesamiento de señales CSI
    """

    def __init__(self, sampling_rate=1000):
        """
        Inicializa el preprocesador

        Args:
            sampling_rate (int): Frecuencia de muestreo en Hz
        """
        self.sampling_rate = sampling_rate
        self.scaler = None
        self.preprocessing_params = {}

    def apply_bandpass_filter(self, data, lowcut=1, highcut=100, order=4):
        """
        Aplica filtro pasa-banda Butterworth

        Args:
            data: Señales CSI [tiempo, antenas, subportadoras] o [tiempo, canales]
            lowcut (float): Frecuencia de corte inferior (Hz)
            highcut (float): Frecuencia de corte superior (Hz)
            order (int): Orden del filtro

        Returns:
            np.ndarray: Datos filtrados
        """
        print(f"Aplicando filtro pasa-banda: {lowcut}-{highcut} Hz, orden {order}")

        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        # Verificar que las frecuencias estén en rango válido
        if low <= 0:
            low = 0.001
        if high >= 1:
            high = 0.999

        b, a = signal.butter(order, [low, high], btype='band')

        filtered_data = np.zeros_like(data)

        if len(data.shape) == 3:
            # Datos 3D [tiempo, antenas, subportadoras]
            for ant in range(data.shape[1]):
                for sub in range(data.shape[2]):
                    filtered_data[:, ant, sub] = signal.filtfilt(b, a, data[:, ant, sub])
        else:
            # Datos 2D [tiempo, canales]
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])

        self.preprocessing_params['bandpass_filter'] = {
            'lowcut': lowcut, 'highcut': highcut, 'order': order
        }

        return filtered_data

    def apply_median_filter(self, data, kernel_size=3):
        """
        Aplica filtro mediano para eliminar picos

        Args:
            data: Datos de entrada
            kernel_size (int): Tamaño del kernel

        Returns:
            np.ndarray: Datos filtrados
        """
        print(f"Aplicando filtro mediano con kernel size: {kernel_size}")

        filtered_data = np.zeros_like(data)

        if len(data.shape) == 3:
            for ant in range(data.shape[1]):
                for sub in range(data.shape[2]):
                    filtered_data[:, ant, sub] = signal.medfilt(data[:, ant, sub], kernel_size)
        else:
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = signal.medfilt(data[:, ch], kernel_size)

        return filtered_data

    def apply_moving_average(self, data, window_size=5):
        """
        Aplica filtro de media móvil

        Args:
            data: Datos de entrada
            window_size (int): Tamaño de la ventana

        Returns:
            np.ndarray: Datos suavizados
        """
        print(f"Aplicando media móvil con ventana: {window_size}")

        filtered_data = np.zeros_like(data)

        if len(data.shape) == 3:
            for ant in range(data.shape[1]):
                for sub in range(data.shape[2]):
                    filtered_data[:, ant, sub] = np.convolve(
                        data[:, ant, sub],
                        np.ones(window_size) / window_size,
                        mode='same'
                    )
        else:
            for ch in range(data.shape[1]):
                filtered_data[:, ch] = np.convolve(
                    data[:, ch],
                    np.ones(window_size) / window_size,
                    mode='same'
                )

        return filtered_data

    def remove_outliers(self, data, method='zscore', threshold=3):
        """
        Elimina outliers usando diferentes métodos

        Args:
            data: Datos de entrada
            method (str): 'zscore', 'iqr'
            threshold (float): Umbral para detección

        Returns:
            tuple: (datos_limpios, mascara_outliers)
        """
        print(f"Eliminando outliers usando método: {method}, umbral: {threshold}")

        clean_data = data.copy()
        outlier_mask = np.zeros_like(data, dtype=bool)

        if method == 'zscore':
            if len(data.shape) == 3:
                for ant in range(data.shape[1]):
                    for sub in range(data.shape[2]):
                        z_scores = np.abs(zscore(data[:, ant, sub]))
                        outliers = z_scores > threshold
                        outlier_mask[:, ant, sub] = outliers

                        # Interpolar outliers
                        if np.any(outliers):
                            clean_data[outliers, ant, sub] = np.interp(
                                np.where(outliers)[0],
                                np.where(~outliers)[0],
                                data[~outliers, ant, sub]
                            )
            else:
                for ch in range(data.shape[1]):
                    z_scores = np.abs(zscore(data[:, ch]))
                    outliers = z_scores > threshold
                    outlier_mask[:, ch] = outliers

                    if np.any(outliers):
                        clean_data[outliers, ch] = np.interp(
                            np.where(outliers)[0],
                            np.where(~outliers)[0],
                            data[~outliers, ch]
                        )

        elif method == 'iqr':
            if len(data.shape) == 3:
                for ant in range(data.shape[1]):
                    for sub in range(data.shape[2]):
                        Q1 = np.percentile(data[:, ant, sub], 25)
                        Q3 = np.percentile(data[:, ant, sub], 75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR

                        outliers = (data[:, ant, sub] < lower_bound) | (data[:, ant, sub] > upper_bound)
                        outlier_mask[:, ant, sub] = outliers

                        if np.any(outliers):
                            clean_data[outliers, ant, sub] = np.interp(
                                np.where(outliers)[0],
                                np.where(~outliers)[0],
                                data[~outliers, ant, sub]
                            )

        outlier_percentage = (np.sum(outlier_mask) / outlier_mask.size) * 100
        print(f"Outliers detectados y corregidos: {outlier_percentage:.2f}%")

        return clean_data, outlier_mask

    def normalize_data(self, data, method='standard'):
        """
        Normaliza los datos usando diferentes métodos

        Args:
            data: Datos de entrada
            method (str): 'standard', 'minmax', 'robust'

        Returns:
            np.ndarray: Datos normalizados
        """
        print(f"Normalizando datos usando método: {method}")

        # Reshape para normalización
        original_shape = data.shape
        if len(data.shape) == 3:
            reshaped_data = data.reshape(data.shape[0], -1)
        else:
            reshaped_data = data

        # Seleccionar scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()

        # Normalizar
        normalized_data = self.scaler.fit_transform(reshaped_data)

        # Volver a forma original
        return normalized_data.reshape(original_shape)

    def create_sliding_windows(self, data, labels, window_size=1000, overlap=0.5, step_size=None):
        """
        Crea ventanas deslizantes de los datos

        Args:
            data: Datos CSI
            labels: Etiquetas correspondientes
            window_size (int): Tamaño de ventana en muestras
            overlap (float): Solapamiento entre ventanas (0-1)
            step_size (int): Tamaño del paso (si None, se calcula desde overlap)

        Returns:
            tuple: (datos_ventaneados, etiquetas_ventaneadas)
        """
        if step_size is None:
            step_size = int(window_size * (1 - overlap))

        print(f"Creando ventanas deslizantes:")
        print(f"  Tamaño de ventana: {window_size} muestras")
        print(f"  Solapamiento: {overlap * 100:.1f}%")
        print(f"  Tamaño del paso: {step_size} muestras")

        # Calcular número de ventanas
        n_windows = (data.shape[0] - window_size) // step_size + 1

        if len(data.shape) == 3:
            # Datos 3D [tiempo, antenas, subportadoras]
            windowed_data = np.zeros((n_windows, window_size, data.shape[1], data.shape[2]))
        else:
            # Datos 2D [tiempo, canales]
            windowed_data = np.zeros((n_windows, window_size, data.shape[1]))

        windowed_labels = np.zeros(n_windows)

        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size

            windowed_data[i] = data[start_idx:end_idx]

            # Para las etiquetas, tomar la etiqueta más frecuente en la ventana
            if labels is not None:
                # Calcular índices de etiquetas correspondientes
                label_start = start_idx // 100  # Ajustar según granularidad original
                label_end = end_idx // 100

                if label_end < len(labels):
                    window_labels = labels[label_start:label_end + 1]
                    if len(window_labels) > 0:
                        # Tomar la etiqueta más frecuente
                        windowed_labels[i] = np.bincount(window_labels.astype(int)).argmax()
                    else:
                        windowed_labels[i] = labels[min(label_start, len(labels) - 1)]
                else:
                    windowed_labels[i] = labels[-1]

        print(f"Ventanas creadas: {n_windows}")
        print(f"Shape final: {windowed_data.shape}")

        return windowed_data, windowed_labels

    def denoise_with_wavelet(self, data, wavelet='db4', threshold_mode='soft'):
        """
        Denoising usando transformada wavelet

        Args:
            data: Datos de entrada
            wavelet (str): Tipo de wavelet
            threshold_mode (str): Modo de thresholding

        Returns:
            np.ndarray: Datos denoised
        """
        try:
            import pywt
        except ImportError:
            print("PyWavelets no está instalado. Usando filtro alternativo.")
            return self.apply_moving_average(data, window_size=3)

        print(f"Aplicando denoising wavelet: {wavelet}, modo: {threshold_mode}")

        denoised_data = np.zeros_like(data)

        if len(data.shape) == 3:
            for ant in range(data.shape[1]):
                for sub in range(data.shape[2]):
                    coeffs = pywt.wavedec(data[:, ant, sub], wavelet, level=4)
                    # Calcular umbral automático
                    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                    threshold = sigma * np.sqrt(2 * np.log(len(data[:, ant, sub])))

                    # Aplicar thresholding
                    coeffs_thresh = list(coeffs)
                    coeffs_thresh[1:] = [pywt.threshold(c, threshold, threshold_mode)
                                         for c in coeffs_thresh[1:]]

                    denoised_data[:, ant, sub] = pywt.waverec(coeffs_thresh, wavelet)
        else:
            for ch in range(data.shape[1]):
                coeffs = pywt.wavedec(data[:, ch], wavelet, level=4)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(data[:, ch])))

                coeffs_thresh = list(coeffs)
                coeffs_thresh[1:] = [pywt.threshold(c, threshold, threshold_mode)
                                     for c in coeffs_thresh[1:]]

                denoised_data[:, ch] = pywt.waverec(coeffs_thresh, wavelet)

        return denoised_data

    def remove_dc_component(self, data):
        """
        Elimina la componente DC (media) de las señales

        Args:
            data: Datos de entrada

        Returns:
            np.ndarray: Datos sin componente DC
        """
        print("Eliminando componente DC...")

        if len(data.shape) == 3:
            # Calcular media a lo largo del eje temporal
            dc_component = np.mean(data, axis=0, keepdims=True)
            return data - dc_component
        else:
            dc_component = np.mean(data, axis=0, keepdims=True)
            return data - dc_component

    def visualize_preprocessing_effects(self, original_data, processed_data,
                                        title="Efectos del Preprocesamiento",
                                        figsize=(15, 10), save_path=None):
        """
        Visualiza los efectos del preprocesamiento

        Args:
            original_data: Datos originales
            processed_data: Datos procesados
            title (str): Título de la figura
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Seleccionar una señal para visualizar
        if len(original_data.shape) == 3:
            original_signal = original_data[:2000, 0, 0]
            processed_signal = processed_data[:2000, 0, 0]
        else:
            original_signal = original_data[:2000, 0]
            processed_signal = processed_data[:2000, 0]

        # Señal original vs procesada
        axes[0, 0].plot(original_signal, label='Original', alpha=0.7, color='blue')
        axes[0, 0].plot(processed_signal, label='Procesada', alpha=0.7, color='red')
        axes[0, 0].set_title('Comparación Temporal')
        axes[0, 0].set_xlabel('Tiempo (muestras)')
        axes[0, 0].set_ylabel('Amplitud')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Espectros de potencia
        freqs_orig, psd_orig = signal.welch(original_signal, fs=self.sampling_rate, nperseg=256)
        freqs_proc, psd_proc = signal.welch(processed_signal, fs=self.sampling_rate, nperseg=256)

        axes[0, 1].semilogy(freqs_orig, psd_orig, label='Original', alpha=0.7, color='blue')
        axes[0, 1].semilogy(freqs_proc, psd_proc, label='Procesada', alpha=0.7, color='red')
        axes[0, 1].set_title('Espectro de Potencia')
        axes[0, 1].set_xlabel('Frecuencia (Hz)')
        axes[0, 1].set_ylabel('PSD')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Histogramas
        axes[1, 0].hist(original_signal, bins=50, alpha=0.7, label='Original',
                        density=True, color='blue')
        axes[1, 0].hist(processed_signal, bins=50, alpha=0.7, label='Procesada',
                        density=True, color='red')
        axes[1, 0].set_title('Distribución de Amplitudes')
        axes[1, 0].set_xlabel('Amplitud')
        axes[1, 0].set_ylabel('Densidad')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Estadísticas
        stats_original = [np.mean(original_signal), np.std(original_signal),
                          np.min(original_signal), np.max(original_signal)]
        stats_processed = [np.mean(processed_signal), np.std(processed_signal),
                           np.min(processed_signal), np.max(processed_signal)]

        x_pos = np.arange(4)
        width = 0.35

        axes[1, 1].bar(x_pos - width / 2, stats_original, width, label='Original', alpha=0.7, color='blue')
        axes[1, 1].bar(x_pos + width / 2, stats_processed, width, label='Procesada', alpha=0.7, color='red')
        axes[1, 1].set_title('Estadísticas Comparativas')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['Media', 'Std', 'Min', 'Max'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()


class CSIPreprocessingPipeline:
    """
    Pipeline completo de preprocesamiento para señales CSI
    """

    def __init__(self, sampling_rate=1000):
        """
        Inicializa el pipeline

        Args:
            sampling_rate (int): Frecuencia de muestreo
        """
        self.preprocessor = CSIPreprocessor(sampling_rate)
        self.steps = []

    def add_step(self, step_name, step_function, **kwargs):
        """
        Añade un paso al pipeline

        Args:
            step_name (str): Nombre del paso
            step_function: Función a ejecutar
            **kwargs: Parámetros para la función
        """
        self.steps.append({
            'name': step_name,
            'function': step_function,
            'params': kwargs
        })

    def fit_transform(self, data, labels=None, visualize=True):
        """
        Ejecuta todo el pipeline de preprocesamiento

        Args:
            data: Datos de entrada
            labels: Etiquetas (opcional)
            visualize (bool): Mostrar visualizaciones

        Returns:
            tuple: (datos_procesados, etiquetas_procesadas)
        """
        print("Iniciando pipeline de preprocesamiento...")
        print(f"Datos originales: {data.shape}")

        processed_data = data.copy()
        original_data = data.copy()

        for step in self.steps:
            print(f"\n--- Ejecutando: {step['name']} ---")

            if step['name'] == 'create_windows' and labels is not None:
                processed_data, labels = step['function'](processed_data, labels, **step['params'])
            else:
                if 'labels' in step['params']:
                    step['params'].pop('labels')
                processed_data = step['function'](processed_data, **step['params'])

        print(f"\nPreprocesamiento completado.")
        print(f"Datos finales: {processed_data.shape}")

        if visualize and len(self.steps) > 0:
            self.preprocessor.visualize_preprocessing_effects(
                original_data, processed_data,
                "Pipeline de Preprocesamiento Completo"
            )

        return processed_data, labels


def analyze_preprocessing_quality(original_data, processed_data, sampling_rate=1000):
    """
    Analiza la calidad del preprocesamiento

    Args:
        original_data: Datos originales
        processed_data: Datos procesados
        sampling_rate (int): Frecuencia de muestreo

    Returns:
        dict: Métricas de calidad
    """
    print("=== ANÁLISIS DE CALIDAD DEL PREPROCESAMIENTO ===")

    # Métricas de calidad
    def calculate_snr(signal):
        signal_power = np.mean(signal ** 2)
        noise_power = np.var(signal - signal.mean())
        return 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Seleccionar señales de muestra
    if len(original_data.shape) == 3:
        orig_sample = original_data[:, 0, 0]
        proc_sample = processed_data[:, 0, 0] if len(processed_data.shape) == 3 else processed_data[:, 0]
    else:
        orig_sample = original_data[:, 0]
        proc_sample = processed_data[:, 0]

    # Limitar tamaño para análisis
    max_samples = min(5000, len(orig_sample), len(proc_sample))
    orig_sample = orig_sample[:max_samples]
    proc_sample = proc_sample[:max_samples]

    # Calcular métricas
    snr_original = calculate_snr(orig_sample)
    snr_processed = calculate_snr(proc_sample)

    # Smoothness (variación total)
    tv_original = np.sum(np.abs(np.diff(orig_sample)))
    tv_processed = np.sum(np.abs(np.diff(proc_sample)))

    # Correlación con original
    correlation = np.corrcoef(orig_sample, proc_sample)[0, 1]

    # Mostrar resultados
    print(f"SNR Original: {snr_original:.2f} dB")
    print(f"SNR Procesada: {snr_processed:.2f} dB")
    print(f"Mejora en SNR: {snr_processed - snr_original:.2f} dB")
    print(f"Variación Total Original: {tv_original:.2f}")
    print(f"Variación Total Procesada: {tv_processed:.2f}")
    print(f"Reducción de Variación: {((tv_original - tv_processed) / tv_original * 100):.1f}%")
    print(f"Correlación con Original: {correlation:.4f}")

    return {
        'snr_improvement': snr_processed - snr_original,
        'variation_reduction': (tv_original - tv_processed) / tv_original * 100,
        'correlation': correlation,
        'snr_original': snr_original,
        'snr_processed': snr_processed
    }


def save_preprocessing_config(pipeline, filepath):
    """
    Guarda la configuración del pipeline de preprocesamiento

    Args:
        pipeline: Pipeline de preprocesamiento
        filepath (str): Ruta del archivo
    """
    import pickle

    config = {
        'steps': pipeline.steps,
        'preprocessing_params': pipeline.preprocessor.preprocessing_params,
        'scaler': pipeline.preprocessor.scaler,
        'sampling_rate': pipeline.preprocessor.sampling_rate
    }

    with open(filepath, 'wb') as f:
        pickle.dump(config, f)

    print(f"Configuración guardada en: {filepath}")


def load_preprocessing_config(filepath):
    """
    Carga la configuración del pipeline de preprocesamiento

    Args:
        filepath (str): Ruta del archivo

    Returns:
        CSIPreprocessingPipeline: Pipeline configurado
    """
    import pickle

    with open(filepath, 'rb') as f:
        config = pickle.load(f)

    pipeline = CSIPreprocessingPipeline(config.get('sampling_rate', 1000))
    pipeline.steps = config['steps']
    pipeline.preprocessor.preprocessing_params = config['preprocessing_params']
    pipeline.preprocessor.scaler = config['scaler']

    print(f"Configuración cargada desde: {filepath}")
    return pipeline


def demo_preprocessing_pipeline():
    """
    Demostración del pipeline de preprocesamiento
    """
    print("=== DEMOSTRACIÓN DEL PIPELINE DE PREPROCESAMIENTO ===")

    # Generar datos sintéticos con ruido
    np.random.seed(42)
    n_samples = 5000
    n_antennas = 3
    n_subcarriers = 8

    # Crear señal base con diferentes patrones
    t = np.linspace(0, n_samples / 1000, n_samples)
    base_signal = (np.sin(2 * np.pi * 2 * t) +
                   0.5 * np.sin(2 * np.pi * 5 * t) +
                   0.3 * np.sin(2 * np.pi * 10 * t))

    # Añadir ruido y artefactos
    noise = 0.3 * np.random.randn(n_samples)
    outliers = np.zeros(n_samples)
    outlier_indices = np.random.choice(n_samples, size=n_samples // 100, replace=False)
    outliers[outlier_indices] = 5 * np.random.randn(len(outlier_indices))

    # Crear datos 3D
    csi_data = np.zeros((n_samples, n_antennas, n_subcarriers))
    for ant in range(n_antennas):
        for sub in range(n_subcarriers):
            variation = 0.1 * np.random.randn(n_samples)
            csi_data[:, ant, sub] = base_signal + noise + outliers + variation

    # Crear etiquetas sintéticas
    labels = np.zeros(n_samples // 100)
    for i in range(len(labels)):
        labels[i] = (i // 10) % 3  # 3 actividades

    print(f"Datos sintéticos creados: {csi_data.shape}")

    # Crear pipeline
    pipeline = CSIPreprocessingPipeline(sampling_rate=1000)

    # Añadir pasos al pipeline
    pipeline.add_step('remove_dc', pipeline.preprocessor.remove_dc_component)
    pipeline.add_step('bandpass_filter', pipeline.preprocessor.apply_bandpass_filter,
                      lowcut=1, highcut=50, order=4)
    pipeline.add_step('remove_outliers', pipeline.preprocessor.remove_outliers,
                      method='zscore', threshold=3)
    pipeline.add_step('median_filter', pipeline.preprocessor.apply_median_filter,
                      kernel_size=3)
    pipeline.add_step('normalize', pipeline.preprocessor.normalize_data,
                      method='standard')
    pipeline.add_step('create_windows', pipeline.preprocessor.create_sliding_windows,
                      window_size=500, overlap=0.5)

    # Ejecutar pipeline
    processed_data, processed_labels = pipeline.fit_transform(csi_data, labels)

    # Analizar calidad
    quality_metrics = analyze_preprocessing_quality(csi_data, processed_data)

    print(f"\nResultados del pipeline:")
    print(f"Datos procesados: {processed_data.shape}")
    print(f"Etiquetas procesadas: {processed_labels.shape}")

    return processed_data, processed_labels, quality_metrics, pipeline


if __name__ == "__main__":
    # Ejecutar demostración
    processed_data, processed_labels, quality_metrics, pipeline = demo_preprocessing_pipeline()

    print("\n=== RESUMEN ===")
    print(f"Mejora en SNR: {quality_metrics['snr_improvement']:.2f} dB")
    print(f"Reducción de variación: {quality_metrics['variation_reduction']:.1f}%")
    print(f"Correlación preservada: {quality_metrics['correlation']:.4f}")

    print("\n✅ Demostración del preprocesamiento completada!")