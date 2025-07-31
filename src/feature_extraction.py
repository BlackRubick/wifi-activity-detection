"""
Módulo de extracción de características de señales CSI
====================================================

Este módulo implementa diferentes técnicas de extracción de características:
- Características estadísticas en dominio temporal
- Características en dominio frecuencial
- Características tiempo-frecuencia (espectrogramas, wavelets)
- Características de energía y correlación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings

warnings.filterwarnings('ignore')


class CSIFeatureExtractor:
    """
    Extractor de características para señales CSI
    """

    def __init__(self, sampling_rate=1000):
        """
        Inicializa el extractor de características

        Args:
            sampling_rate (int): Frecuencia de muestreo en Hz
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []

    def extract_temporal_features(self, data):
        """
        Extrae características estadísticas en dominio temporal

        Args:
            data: Array con datos CSI [ventanas, tiempo, antenas, subportadoras] o [tiempo, antenas, subportadoras]

        Returns:
            np.ndarray: Características temporales extraídas
        """
        print("Extrayendo características temporales...")

        features = []
        feature_names = []

        # Determinar formato de datos
        if len(data.shape) == 4:  # [ventanas, tiempo, antenas, subportadoras]
            n_windows = data.shape[0]
            process_windowed = True
        else:  # [tiempo, antenas, subportadoras] - una sola ventana
            n_windows = 1
            process_windowed = False
            data = data[np.newaxis, ...]  # Añadir dimensión de ventana

        for window_idx in range(n_windows):
            window_features = []
            window_data = data[window_idx]  # [tiempo, antenas, subportadoras]

            # Iterar sobre antenas y subportadoras
            for ant in range(window_data.shape[1]):
                for sub in range(window_data.shape[2]):
                    signal_data = window_data[:, ant, sub]

                    # Características básicas
                    mean_val = np.mean(signal_data)
                    std_val = np.std(signal_data)
                    var_val = np.var(signal_data)
                    median_val = np.median(signal_data)

                    # Momentos estadísticos
                    skewness = stats.skew(signal_data)
                    kurtosis = stats.kurtosis(signal_data)

                    # Características de distribución
                    min_val = np.min(signal_data)
                    max_val = np.max(signal_data)
                    range_val = max_val - min_val

                    # Percentiles
                    q25 = np.percentile(signal_data, 25)
                    q75 = np.percentile(signal_data, 75)
                    iqr = q75 - q25

                    # Características de energía
                    energy = np.sum(signal_data ** 2)
                    rms = np.sqrt(np.mean(signal_data ** 2))

                    # Características de variabilidad
                    mad = np.mean(np.abs(signal_data - mean_val))  # Mean Absolute Deviation
                    cv = std_val / (abs(mean_val) + 1e-10)  # Coefficient of Variation

                    # Zero crossing rate
                    zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
                    zcr = zero_crossings / len(signal_data)

                    # Peak-to-peak
                    peak_to_peak = np.ptp(signal_data)

                    # Crest factor
                    crest_factor = max_val / (rms + 1e-10)

                    window_features.extend([
                        mean_val, std_val, var_val, median_val,
                        skewness, kurtosis, min_val, max_val, range_val,
                        q25, q75, iqr, energy, rms, mad, cv, zcr,
                        peak_to_peak, crest_factor
                    ])

                    # Nombres de características (solo primera iteración)
                    if window_idx == 0:
                        base_names = [
                            'mean', 'std', 'var', 'median',
                            'skewness', 'kurtosis', 'min', 'max', 'range',
                            'q25', 'q75', 'iqr', 'energy', 'rms', 'mad', 'cv', 'zcr',
                            'peak_to_peak', 'crest_factor'
                        ]
                        for name in base_names:
                            feature_names.append(f'temporal_{name}_ant{ant}_sub{sub}')

            features.append(window_features)

        if not hasattr(self, 'temporal_feature_names'):
            self.temporal_feature_names = feature_names

        return np.array(features)

    def extract_frequency_features(self, data):
        """
        Extrae características en dominio frecuencial

        Args:
            data: Array con datos CSI

        Returns:
            np.ndarray: Características frecuenciales extraídas
        """
        print("Extrayendo características frecuenciales...")

        features = []
        feature_names = []

        # Determinar formato de datos
        if len(data.shape) == 4:  # [ventanas, tiempo, antenas, subportadoras]
            n_windows = data.shape[0]
            process_windowed = True
        else:  # [tiempo, antenas, subportadoras] - una sola ventana
            n_windows = 1
            process_windowed = False
            data = data[np.newaxis, ...]

        for window_idx in range(n_windows):
            window_features = []
            window_data = data[window_idx]

            for ant in range(window_data.shape[1]):
                for sub in range(window_data.shape[2]):
                    signal_data = window_data[:, ant, sub]

                    # FFT
                    fft_vals = fft(signal_data)
                    freqs = fftfreq(len(signal_data), 1 / self.sampling_rate)

                    # Solo frecuencias positivas
                    positive_freqs = freqs[:len(freqs) // 2]
                    positive_fft = np.abs(fft_vals[:len(fft_vals) // 2])

                    # PSD usando Welch
                    freqs_welch, psd = signal.welch(signal_data, fs=self.sampling_rate,
                                                    nperseg=min(256, len(signal_data) // 4))

                    # Características espectrales
                    # 1. Frecuencia dominante
                    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Evitar DC
                    dominant_freq = freqs_welch[dominant_freq_idx]

                    # 2. Potencia en bandas de frecuencia
                    def power_in_band(freqs, psd, low, high):
                        mask = (freqs >= low) & (freqs <= high)
                        return np.sum(psd[mask]) if np.any(mask) else 0

                    power_delta = power_in_band(freqs_welch, psd, 0.5, 4)  # Delta
                    power_theta = power_in_band(freqs_welch, psd, 4, 8)  # Theta
                    power_alpha = power_in_band(freqs_welch, psd, 8, 13)  # Alpha
                    power_beta = power_in_band(freqs_welch, psd, 13, 30)  # Beta
                    power_gamma = power_in_band(freqs_welch, psd, 30, 50)  # Gamma

                    total_power = np.sum(psd)

                    # 3. Centroide espectral
                    spectral_centroid = np.sum(psd * freqs_welch) / (np.sum(psd) + 1e-10)

                    # 4. Ancho de banda espectral
                    spectral_spread = np.sqrt(np.sum(((freqs_welch - spectral_centroid) ** 2) * psd) /
                                              (np.sum(psd) + 1e-10))

                    # 5. Rolloff espectral (95% de la energía)
                    cumulative_psd = np.cumsum(psd)
                    rolloff_idx = np.where(cumulative_psd >= 0.95 * cumulative_psd[-1])[0]
                    spectral_rolloff = freqs_welch[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs_welch[-1]

                    # 6. Entropía espectral
                    psd_normalized = psd / (np.sum(psd) + 1e-10)
                    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10))

                    # 7. Planicidad espectral
                    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
                    arithmetic_mean = np.mean(psd)
                    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

                    # 8. Flux espectral (cambio en el espectro)
                    if len(psd) > 1:
                        spectral_flux = np.sum(np.diff(psd) ** 2)
                    else:
                        spectral_flux = 0

                    window_features.extend([
                        dominant_freq, power_delta, power_theta, power_alpha,
                        power_beta, power_gamma, total_power, spectral_centroid,
                        spectral_spread, spectral_rolloff, spectral_entropy,
                        spectral_flatness, spectral_flux
                    ])

                    # Nombres de características (solo primera iteración)
                    if window_idx == 0:
                        freq_names = [
                            'dominant_freq', 'power_delta', 'power_theta', 'power_alpha',
                            'power_beta', 'power_gamma', 'total_power', 'spectral_centroid',
                            'spectral_spread', 'spectral_rolloff', 'spectral_entropy',
                            'spectral_flatness', 'spectral_flux'
                        ]
                        for name in freq_names:
                            feature_names.append(f'freq_{name}_ant{ant}_sub{sub}')

            features.append(window_features)

        if not hasattr(self, 'frequency_feature_names'):
            self.frequency_feature_names = feature_names

        return np.array(features)

    def extract_timefrequency_features(self, data):
        """
        Extrae características tiempo-frecuencia usando espectrogramas

        Args:
            data: Array con datos CSI

        Returns:
            np.ndarray: Características tiempo-frecuencia extraídas
        """
        print("Extrayendo características tiempo-frecuencia...")

        features = []
        feature_names = []

        # Determinar formato de datos
        if len(data.shape) == 4:  # [ventanas, tiempo, antenas, subportadoras]
            n_windows = data.shape[0]
            process_windowed = True
        else:  # [tiempo, antenas, subportadoras] - una sola ventana
            n_windows = 1
            process_windowed = False
            data = data[np.newaxis, ...]

        for window_idx in range(n_windows):
            window_features = []
            window_data = data[window_idx]

            for ant in range(window_data.shape[1]):
                for sub in range(window_data.shape[2]):
                    signal_data = window_data[:, ant, sub]

                    # Crear espectrograma
                    f, t, Sxx = signal.spectrogram(signal_data, fs=self.sampling_rate,
                                                   nperseg=64, noverlap=32)

                    # Características del espectrograma
                    # 1. Energía por bandas de frecuencia
                    n_freq_bands = 5
                    freq_band_size = len(f) // n_freq_bands

                    for band in range(n_freq_bands):
                        start_idx = band * freq_band_size
                        end_idx = min((band + 1) * freq_band_size, len(f))
                        band_energy = np.sum(Sxx[start_idx:end_idx, :])
                        window_features.append(band_energy)

                        if window_idx == 0:
                            feature_names.append(f'spectrogram_band{band}_energy_ant{ant}_sub{sub}')

                    # 2. Centroide temporal del espectrograma
                    temporal_centroid = np.sum(np.sum(Sxx, axis=0) * t) / (np.sum(Sxx) + 1e-10)
                    window_features.append(temporal_centroid)

                    # 3. Dispersión temporal
                    temporal_spread = np.sqrt(np.sum(np.sum(Sxx, axis=0) * (t - temporal_centroid) ** 2) /
                                              (np.sum(Sxx) + 1e-10))
                    window_features.append(temporal_spread)

                    # 4. Máximo valor del espectrograma
                    max_spectrogram = np.max(Sxx)
                    window_features.append(max_spectrogram)

                    # 5. Entropía del espectrograma
                    Sxx_normalized = Sxx / (np.sum(Sxx) + 1e-10)
                    spectrogram_entropy = -np.sum(Sxx_normalized * np.log(Sxx_normalized + 1e-10))
                    window_features.append(spectrogram_entropy)

                    # 6. Variabilidad temporal del espectrograma
                    temporal_var = np.var(np.sum(Sxx, axis=0))
                    window_features.append(temporal_var)

                    # 7. Variabilidad frecuencial del espectrograma
                    freq_var = np.var(np.sum(Sxx, axis=1))
                    window_features.append(freq_var)

                    if window_idx == 0:
                        tf_names = ['temporal_centroid', 'temporal_spread', 'max_spectrogram',
                                    'spectrogram_entropy', 'temporal_var', 'freq_var']
                        for name in tf_names:
                            feature_names.append(f'timefreq_{name}_ant{ant}_sub{sub}')

            features.append(window_features)

        if not hasattr(self, 'timefrequency_feature_names'):
            self.timefrequency_feature_names = feature_names

        return np.array(features)

    def extract_correlation_features(self, data):
        """
        Extrae características de correlación entre antenas y subportadoras

        Args:
            data: Array con datos CSI

        Returns:
            np.ndarray: Características de correlación extraídas
        """
        print("Extrayendo características de correlación...")

        features = []
        feature_names = []

        # Determinar formato de datos
        if len(data.shape) == 4:  # [ventanas, tiempo, antenas, subportadoras]
            n_windows = data.shape[0]
            process_windowed = True
        else:  # [tiempo, antenas, subportadoras] - una sola ventana
            n_windows = 1
            process_windowed = False
            data = data[np.newaxis, ...]

        for window_idx in range(n_windows):
            window_features = []
            window_data = data[window_idx]  # [tiempo, antenas, subportadoras]

            # Correlación entre antenas (para cada subportadora)
            for sub in range(window_data.shape[2]):
                antenna_data = window_data[:, :, sub].T  # [antenas, tiempo]
                if antenna_data.shape[0] > 1:
                    corr_matrix = np.corrcoef(antenna_data)

                    # Extraer valores únicos de la matriz de correlación (triángulo superior)
                    n_antennas = corr_matrix.shape[0]
                    for i in range(n_antennas):
                        for j in range(i + 1, n_antennas):
                            window_features.append(corr_matrix[i, j])
                            if window_idx == 0:
                                feature_names.append(f'corr_ant{i}_ant{j}_sub{sub}')

            # Correlación entre subportadoras (para cada antena) - muestreada
            for ant in range(window_data.shape[1]):
                subcarrier_data = window_data[:, ant, :].T  # [subportadoras, tiempo]
                if subcarrier_data.shape[0] > 1:
                    corr_matrix = np.corrcoef(subcarrier_data)

                    # Extraer algunos valores representativos (no todos para evitar muchas features)
                    n_subcarriers = corr_matrix.shape[0]
                    step = max(1, n_subcarriers // 5)  # Tomar cada 5ta correlación

                    for i in range(0, n_subcarriers, step):
                        for j in range(i + step, n_subcarriers, step):
                            if j < n_subcarriers:
                                window_features.append(corr_matrix[i, j])
                                if window_idx == 0:
                                    feature_names.append(f'corr_sub{i}_sub{j}_ant{ant}')

            # Características estadísticas de las correlaciones
            all_antenna_corrs = []
            for sub in range(window_data.shape[2]):
                antenna_data = window_data[:, :, sub].T
                if antenna_data.shape[0] > 1:
                    corr_matrix = np.corrcoef(antenna_data)
                    # Extraer triángulo superior
                    triu_indices = np.triu_indices_from(corr_matrix, k=1)
                    all_antenna_corrs.extend(corr_matrix[triu_indices])

            if all_antenna_corrs:
                corr_mean = np.mean(all_antenna_corrs)
                corr_std = np.std(all_antenna_corrs)
                corr_max = np.max(all_antenna_corrs)
                corr_min = np.min(all_antenna_corrs)

                window_features.extend([corr_mean, corr_std, corr_max, corr_min])

                if window_idx == 0:
                    corr_stat_names = ['corr_mean', 'corr_std', 'corr_max', 'corr_min']
                    feature_names.extend(corr_stat_names)

            features.append(window_features)

        if not hasattr(self, 'correlation_feature_names'):
            self.correlation_feature_names = feature_names

        return np.array(features)

    def extract_wavelet_features(self, data, wavelet='db4', levels=4):
        """
        Extrae características usando transformada wavelet

        Args:
            data: Array con datos CSI
            wavelet (str): Tipo de wavelet
            levels (int): Niveles de descomposición

        Returns:
            np.ndarray: Características wavelet extraídas
        """
        print(f"Extrayendo características wavelet ({wavelet}, {levels} niveles)...")

        try:
            import pywt
        except ImportError:
            print("PyWavelets no está disponible. Saltando características wavelet.")
            return np.array([])

        features = []
        feature_names = []

        # Determinar formato de datos
        if len(data.shape) == 4:  # [ventanas, tiempo, antenas, subportadoras]
            n_windows = data.shape[0]
            process_windowed = True
        else:  # [tiempo, antenas, subportadoras] - una sola ventana
            n_windows = 1
            process_windowed = False
            data = data[np.newaxis, ...]

        for window_idx in range(n_windows):
            window_features = []
            window_data = data[window_idx]

            for ant in range(window_data.shape[1]):
                for sub in range(window_data.shape[2]):
                    signal_data = window_data[:, ant, sub]

                    # Descomposición wavelet
                    coeffs = pywt.wavedec(signal_data, wavelet, level=levels)

                    # Características de cada nivel
                    for level, coeff in enumerate(coeffs):
                        # Estadísticas de los coeficientes
                        mean_coeff = np.mean(np.abs(coeff))
                        std_coeff = np.std(coeff)
                        energy_coeff = np.sum(coeff ** 2)
                        max_coeff = np.max(np.abs(coeff))

                        window_features.extend([mean_coeff, std_coeff, energy_coeff, max_coeff])

                        if window_idx == 0:
                            coeff_names = ['mean', 'std', 'energy', 'max']
                            for name in coeff_names:
                                feature_names.append(f'wavelet_level{level}_{name}_ant{ant}_sub{sub}')

            features.append(window_features)

        if not hasattr(self, 'wavelet_feature_names'):
            self.wavelet_feature_names = feature_names

        return np.array(features)

    def extract_all_features(self, data):
        """
        Extrae todas las características disponibles

        Args:
            data: Array con datos CSI

        Returns:
            np.ndarray: Todas las características combinadas
        """
        print("Extrayendo todas las características...")
        all_features = []
        all_feature_names = []

        # Características temporales
        temporal_features = self.extract_temporal_features(data)
        if temporal_features.size > 0:
            all_features.append(temporal_features)
            all_feature_names.extend(self.temporal_feature_names)

        # Características frecuenciales
        frequency_features = self.extract_frequency_features(data)
        if frequency_features.size > 0:
            all_features.append(frequency_features)
            all_feature_names.extend(self.frequency_feature_names)

        # Características tiempo-frecuencia
        timefreq_features = self.extract_timefrequency_features(data)
        if timefreq_features.size > 0:
            all_features.append(timefreq_features)
            all_feature_names.extend(self.timefrequency_feature_names)

        # Características de correlación
        correlation_features = self.extract_correlation_features(data)
        if correlation_features.size > 0:
            all_features.append(correlation_features)
            all_feature_names.extend(self.correlation_feature_names)

        # Características wavelet
        wavelet_features = self.extract_wavelet_features(data)
        if wavelet_features.size > 0:
            all_features.append(wavelet_features)
            all_feature_names.extend(self.wavelet_feature_names)

        # Concatenar todas las características
        if all_features:
            combined_features = np.concatenate(all_features, axis=1)
            self.feature_names = all_feature_names

            print(f"Características extraídas: {combined_features.shape}")
            print(f"Tipos de características:")
            print(f"  - Temporales: {len(getattr(self, 'temporal_feature_names', []))}")
            print(f"  - Frecuenciales: {len(getattr(self, 'frequency_feature_names', []))}")
            print(f"  - Tiempo-frecuencia: {len(getattr(self, 'timefrequency_feature_names', []))}")
            print(f"  - Correlación: {len(getattr(self, 'correlation_feature_names', []))}")
            print(f"  - Wavelet: {len(getattr(self, 'wavelet_feature_names', []))}")

            return combined_features
        else:
            return np.array([])


def analyze_feature_importance(features, labels, feature_names, method='f_classif', k=50):
    """
    Analiza la importancia de las características - VERSIÓN CORREGIDA PARA VALORES INFINITOS
    """
    print(f"Analizando importancia de características usando {method}...")

    # ======================================
    # FIX: Limpiar datos antes del análisis
    # ======================================
    print(f"📊 Datos originales: {features.shape}")
    print(f"   NaN: {np.sum(np.isnan(features))}")
    print(f"   Inf: {np.sum(np.isinf(features))}")

    # Limpiar valores infinitos y NaN
    features_clean = features.copy()

    # Reemplazar infinitos con valores finitos
    inf_mask = np.isinf(features_clean)
    if np.any(inf_mask):
        print(f"⚠️ Encontrados {np.sum(inf_mask)} valores infinitos - reemplazando...")

        # Para cada columna, reemplazar inf con el valor máximo finito * 2
        for col in range(features_clean.shape[1]):
            col_data = features_clean[:, col]
            if np.any(np.isinf(col_data)):
                finite_values = col_data[np.isfinite(col_data)]
                if len(finite_values) > 0:
                    max_finite = np.max(finite_values)
                    features_clean[np.isinf(col_data), col] = max_finite * 2
                else:
                    features_clean[np.isinf(col_data), col] = 1.0

    # Reemplazar NaN con mediana
    nan_mask = np.isnan(features_clean)
    if np.any(nan_mask):
        print(f"⚠️ Encontrados {np.sum(nan_mask)} valores NaN - reemplazando...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        features_clean = imputer.fit_transform(features_clean)

    print(f"✅ Datos limpios: NaN={np.sum(np.isnan(features_clean))}, Inf={np.sum(np.isinf(features_clean))}")

    # Continuar con el análisis usando datos limpios
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        print(f"Método {method} no reconocido. Usando f_classif.")
        selector = SelectKBest(score_func=f_classif, k=k)

    # Seleccionar características
    features_selected = selector.fit_transform(features_clean, labels)

    # Obtener scores e indices
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features_names = [feature_names[i] for i in selected_indices]

    # ======================================
    # FIX: Limpiar scores para visualización
    # ======================================
    # Reemplazar scores infinitos
    scores_clean = scores.copy()
    inf_scores = np.isinf(scores_clean)
    if np.any(inf_scores):
        print(f"⚠️ {np.sum(inf_scores)} scores infinitos encontrados - reemplazando...")
        finite_scores = scores_clean[np.isfinite(scores_clean)]
        if len(finite_scores) > 0:
            max_finite_score = np.max(finite_scores)
            scores_clean[inf_scores] = max_finite_score * 2
        else:
            scores_clean[inf_scores] = 1.0

    # Crear DataFrame para análisis
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'score': scores_clean,  # Usar scores limpios
        'selected': selector.get_support()
    }).sort_values('score', ascending=False)

    # Visualizar top características
    plt.figure(figsize=(12, 8))

    # Top 20 características
    top_features = importance_df.head(20)
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(len(top_features)), top_features['score'])
    plt.xticks(range(len(top_features)), [name[:30] + '...' if len(name) > 30 else name
                                          for name in top_features['feature']],
               rotation=45, ha='right')
    plt.title(f'Top 20 Características Más Importantes ({method})')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)

    # Colorear barras seleccionadas
    for i, (_, row) in enumerate(top_features.iterrows()):
        if row['selected']:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('lightblue')

    # Distribución de scores - CON PROTECCIÓN ADICIONAL
    plt.subplot(2, 1, 2)
    try:
        # Filtrar scores finitos para el histograma
        finite_scores_for_hist = importance_df['score'][np.isfinite(importance_df['score'])]

        if len(finite_scores_for_hist) > 0:
            plt.hist(finite_scores_for_hist, bins=50, alpha=0.7, edgecolor='black')

            # Threshold line
            if k <= len(importance_df):
                threshold_score = importance_df.iloc[k - 1]['score']
                if np.isfinite(threshold_score):
                    plt.axvline(threshold_score, color='red', linestyle='--',
                                label=f'Umbral top-{k}')
                    plt.legend()
        else:
            plt.text(0.5, 0.5, 'No hay scores finitos para mostrar',
                     ha='center', va='center', transform=plt.gca().transAxes)

        plt.title('Distribución de Scores de Importancia')
        plt.xlabel('Score')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)

    except Exception as e:
        print(f"⚠️ Error creando histograma: {e}")
        plt.text(0.5, 0.5, f'Error en visualización: {str(e)[:50]}...',
                 ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    print(f"Características seleccionadas: {len(selected_features_names)}")
    print(f"Reducción dimensional: {features.shape[1]} -> {features_selected.shape[1]}")

    return features_selected, selected_features_names, importance_df


def apply_pca_analysis(features, n_components=0.95):
    """
    Aplica análisis de componentes principales

    Args:
        features: Array de características
        n_components: Número de componentes o varianza a explicar

    Returns:
        tuple: (características_pca, objeto_pca)
    """
    print(f"Aplicando PCA (componentes={n_components})...")

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)

    # Análisis de varianza explicada
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada por PCA')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    n_components_plot = min(20, len(pca.explained_variance_ratio_))
    plt.bar(range(n_components_plot), pca.explained_variance_ratio_[:n_components_plot])
    plt.xlabel('Componente Principal')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza por Componente (Top 20)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Componentes principales: {features_pca.shape[1]}")
    print(f"Varianza explicada total: {np.sum(pca.explained_variance_ratio_):.3f}")

    return features_pca, pca


def visualize_feature_correlations(features, feature_names, sample_size=100, figsize=(12, 10)):
    """
    Visualiza correlaciones entre características

    Args:
        features: Array de características
        feature_names: Nombres de las características
        sample_size (int): Número máximo de características a mostrar
        figsize (tuple): Tamaño de la figura

    Returns:
        np.ndarray: Matriz de correlación
    """
    print("Visualizando correlaciones entre características...")

    # Limitar número de características para visualización
    if len(feature_names) > sample_size:
        # Seleccionar aleatoriamente características para visualizar
        np.random.seed(42)
        selected_indices = np.random.choice(len(feature_names), sample_size, replace=False)
        features_sample = features[:, selected_indices]
        names_sample = [feature_names[i] for i in selected_indices]
    else:
        features_sample = features
        names_sample = feature_names

    # Calcular matriz de correlación
    corr_matrix = np.corrcoef(features_sample.T)

    # Visualizar
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Matriz de Correlación entre Características')
    plt.tight_layout()
    plt.show()

    # Encontrar características altamente correlacionadas
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > 0.8:
                high_corr_pairs.append((names_sample[i], names_sample[j], corr_matrix[i, j]))

    if high_corr_pairs:
        print(f"\nCaracterísticas altamente correlacionadas (|r| > 0.8):")
        for feat1, feat2, corr in high_corr_pairs[:10]:  # Top 10
            print(f"  {feat1[:50]} <-> {feat2[:50]}: {corr:.3f}")

    return corr_matrix


def plot_features_by_activity(features, labels, feature_names, activity_names, n_features=6):
    """
    Visualiza características agrupadas por actividad

    Args:
        features: Array de características
        labels: Etiquetas de actividad
        feature_names: Nombres de las características
        activity_names: Nombres de las actividades
        n_features (int): Número de características a mostrar
    """
    n_features = min(n_features, len(feature_names))
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]

        # Crear boxplot por actividad
        data_by_activity = []
        activity_labels = []

        for activity_idx in range(len(activity_names)):
            mask = labels == activity_idx
            if np.any(mask):
                data_by_activity.append(features[mask, i])
                activity_labels.append(activity_names[activity_idx])

        if data_by_activity:
            ax.boxplot(data_by_activity, labels=activity_labels)
            feature_name = feature_names[i]
            ax.set_title(f'{feature_name[:40]}...', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

    # Ocultar axes vacíos
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Distribución de Características por Actividad', y=1.02)
    plt.show()


def create_feature_summary_report(extractor, features, labels, activity_names):
    """
    Crea un reporte resumen de las características extraídas

    Args:
        extractor: Objeto extractor de características
        features: Array de características
        labels: Etiquetas
        activity_names: Nombres de las actividades
    """
    print("\n" + "=" * 60)
    print("REPORTE RESUMEN DE CARACTERÍSTICAS")
    print("=" * 60)

    # Estadísticas generales
    print(f"Número total de características: {features.shape[1]}")
    print(f"Número de muestras: {features.shape[0]}")
    print(f"Actividades: {', '.join(activity_names)}")
    print(f"Distribución de clases: {dict(zip(activity_names, np.bincount(labels.astype(int))))}")

    # Análisis por tipo de característica
    print("\nDISTRIBUCIÓN POR TIPO DE CARACTERÍSTICA:")
    feature_types = {}
    for name in extractor.feature_names:
        feature_type = name.split('_')[0]
        feature_types[feature_type] = feature_types.get(feature_type, 0) + 1

    for ftype, count in feature_types.items():
        percentage = (count / len(extractor.feature_names)) * 100
        print(f"  {ftype.capitalize()}: {count} ({percentage:.1f}%)")

    # Estadísticas de las características
    print(f"\nESTADÍSTICAS DE CARACTERÍSTICAS:")
    print(f"  Media general: {np.mean(features):.6f}")
    print(f"  Desviación estándar: {np.std(features):.6f}")
    print(f"  Rango: [{np.min(features):.6f}, {np.max(features):.6f}]")

    # Verificar valores NaN o infinitos
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))

    if nan_count > 0:
        print(f"  ⚠️ Valores NaN detectados: {nan_count}")
    if inf_count > 0:
        print(f"  ⚠️ Valores infinitos detectados: {inf_count}")

    # Recomendaciones
    print(f"\nRECOMENDACIONES:")
    n_features = features.shape[1]
    n_samples = features.shape[0]

    if n_features > 1000:
        print("  - Considera usar selección de características o PCA (muchas características)")
    elif n_features < 50:
        print("  - Podrías beneficiarte de extraer más características")
    else:
        print("  - Número de características adecuado")

    if n_samples < n_features:
        print("  - ⚠️ Más características que muestras: riesgo de overfitting")

    print("\n" + "=" * 60)


def save_features_and_metadata(features, labels, feature_names, activity_names, filepath):
    """
    Guarda características y metadatos

    Args:
        features: Array de características
        labels: Etiquetas
        feature_names: Nombres de las características
        activity_names: Nombres de las actividades
        filepath (str): Ruta del archivo
    """
    import pickle

    data_to_save = {
        'features': features,
        'labels': labels,
        'feature_names': feature_names,
        'activity_names': activity_names,
        'shape': features.shape,
        'timestamp': pd.Timestamp.now().isoformat(),
        'statistics': {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        }
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Características guardadas en: {filepath}")


def load_features_and_metadata(filepath):
    """
    Carga características y metadatos

    Args:
        filepath (str): Ruta del archivo

    Returns:
        tuple: (características, etiquetas, nombres_características, nombres_actividades)
    """
    import pickle

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    print(f"Características cargadas desde: {filepath}")
    print(f"Shape: {data['shape']}")
    print(f"Guardado en: {data['timestamp']}")

    return data['features'], data['labels'], data['feature_names'], data['activity_names']


def demo_feature_extraction():
    """
    Demostración completa de extracción de características
    """
    print("=== DEMOSTRACIÓN DE EXTRACCIÓN DE CARACTERÍSTICAS ===")

    # Generar datos sintéticos
    np.random.seed(42)
    n_windows = 100
    window_size = 500
    n_antennas = 3
    n_subcarriers = 8

    # Crear datos sintéticos con patrones diferentes por actividad
    activities = ['caminar', 'correr', 'sentado']
    n_activities = len(activities)

    data = np.zeros((n_windows, window_size, n_antennas, n_subcarriers))
    labels = np.zeros(n_windows)

    for window in range(n_windows):
        activity_idx = window % n_activities
        labels[window] = activity_idx

        # Crear patrones específicos por actividad
        for ant in range(n_antennas):
            for sub in range(n_subcarriers):
                t = np.linspace(0, window_size / 1000, window_size)

                if activity_idx == 0:  # caminar
                    pattern = (np.sin(2 * np.pi * 2 * t) +
                               0.5 * np.sin(2 * np.pi * 4 * t) +
                               0.3 * np.random.randn(window_size))
                elif activity_idx == 1:  # correr
                    pattern = (2 * np.sin(2 * np.pi * 4 * t) +
                               0.8 * np.sin(2 * np.pi * 8 * t) +
                               0.4 * np.random.randn(window_size))
                else:  # sentado
                    pattern = (0.1 * np.sin(2 * np.pi * 0.5 * t) +
                               0.05 * np.random.randn(window_size))

                # Añadir variaciones por antena y subportadora
                variation = 0.1 * (ant + 1) * (sub + 1) * np.random.randn(window_size)
                data[window, :, ant, sub] = pattern + variation

    print(f"Datos sintéticos creados: {data.shape}")
    print(f"Distribución de etiquetas: {np.bincount(labels.astype(int))}")

    # Crear extractor de características
    extractor = CSIFeatureExtractor(sampling_rate=1000)

    # Extraer todas las características
    features = extractor.extract_all_features(data)

    if features.size == 0:
        print("Error: No se pudieron extraer características")
        return

    print(f"\nCaracterísticas extraídas: {features.shape}")

    # Análisis de importancia de características
    print("\n--- ANÁLISIS DE IMPORTANCIA ---")
    features_selected, selected_names, importance_df = analyze_feature_importance(
        features, labels, extractor.feature_names, method='f_classif', k=30
    )

    # Análisis PCA
    print("\n--- ANÁLISIS PCA ---")
    features_pca, pca = apply_pca_analysis(features, n_components=0.95)

    # Visualización de correlaciones
    print("\n--- ANÁLISIS DE CORRELACIONES ---")
    corr_matrix = visualize_feature_correlations(features, extractor.feature_names, sample_size=50)

    # Comparar diferentes representaciones
    print("\n--- COMPARACIÓN DE REPRESENTACIONES ---")
    print(f"Características originales: {features.shape}")
    print(f"Características seleccionadas: {features_selected.shape}")
    print(f"Características PCA: {features_pca.shape}")

    # Visualizar algunas características por actividad
    plot_features_by_activity(features_selected, labels, selected_names[:6], activities)

    # Crear reporte resumen
    create_feature_summary_report(extractor, features, labels, activities)

    return {
        'extractor': extractor,
        'features_original': features,
        'features_selected': features_selected,
        'features_pca': features_pca,
        'labels': labels,
        'feature_names': extractor.feature_names,
        'selected_names': selected_names,
        'importance_df': importance_df,
        'activities': activities,
        'pca': pca
    }


if __name__ == "__main__":
    # Ejecutar demostración completa
    results = demo_feature_extraction()

    if results:
        print(f"\n✅ Extracción de características completada exitosamente!")
        print(f"📊 {results['features_original'].shape[1]} características originales")
        print(f"🎯 {results['features_selected'].shape[1]} características seleccionadas")
        print(f"📈 {results['features_pca'].shape[1]} componentes principales")

        # Ejemplo de guardado
        # save_features_and_metadata(
        #     results['features_selected'],
        #     results['labels'],
        #     results['selected_names'],
        #     results['activities'],
        #     'results/features_selected.pkl'
        # )