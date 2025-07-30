"""
Módulo de carga y análisis de datos del dataset WI-MIR
=====================================================

Este módulo contiene todas las funcionalidades para cargar, explorar y analizar
los datos del dataset WI-MIR para detección de actividades humanas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
import h5py
import os
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


class WiFiDataLoader:
    """
    Cargador y explorador de datos para el dataset WI-MIR
    """

    def __init__(self, dataset_path=None):
        """
        Inicializa el cargador de datos

        Args:
            dataset_path (str): Ruta al archivo del dataset
        """
        self.dataset_path = dataset_path
        self.raw_data = None
        self.csi_data = None
        self.labels = None
        self.activities = None
        self.metadata = {}

    def load_wimir_data(self, file_path=None):
        """
        Carga datos del dataset WI-MIR desde archivo .mat

        Args:
            file_path (str): Ruta al archivo (opcional)

        Returns:
            dict: Diccionario con datos CSI y metadatos
        """
        if file_path is None:
            file_path = self.dataset_path

        if file_path is None:
            raise ValueError("Se debe proporcionar una ruta al archivo")

        try:
            # Cargar archivo .mat
            print(f"Cargando datos desde: {file_path}")
            data = loadmat(file_path)

            # Filtrar claves privadas de MATLAB
            clean_data = {k: v for k, v in data.items() if not k.startswith('__')}

            self.raw_data = clean_data

            print(f"Archivo cargado exitosamente")
            print(f"Claves disponibles: {list(clean_data.keys())}")

            # Extraer datos CSI y etiquetas automáticamente
            self._extract_csi_and_labels()

            return clean_data

        except Exception as e:
            print(f"Error al cargar archivo: {e}")
            raise

    def _extract_csi_and_labels(self):
        """
        Extrae datos CSI y etiquetas de estructuras MATLAB
        VERSIÓN PARA ESTRUCTURAS CON CAMPOS 'csi' y 'MPIs_label'
        """
        if self.raw_data is None:
            return

        print("🔍 Extrayendo datos de estructuras MATLAB...")

        if 'Raw_Cell_Matrix' not in self.raw_data:
            print("❌ No se encontró Raw_Cell_Matrix")
            return

        cell_matrix = self.raw_data['Raw_Cell_Matrix']
        print(f"📦 Cell matrix shape: {cell_matrix.shape}")

        all_csi_data = []
        all_labels = []

        print("🔄 Procesando estructuras...")

        for i in range(min(1000, cell_matrix.size)):  # Procesar hasta 1000 estructuras
            try:
                # Extraer estructura
                if cell_matrix.ndim == 2:
                    struct = cell_matrix[i, 0]
                else:
                    struct = cell_matrix.flat[i]

                if not isinstance(struct, np.ndarray) or struct.size == 0:
                    continue

                # La estructura debería tener un solo elemento
                struct_data = struct[0, 0] if struct.ndim == 2 else struct.flat[0]

                # Verificar que tiene los campos necesarios
                if hasattr(struct_data, 'dtype') and 'csi' in struct_data.dtype.names:
                    csi_field = struct_data['csi']

                    # Extraer datos CSI reales
                    if isinstance(csi_field, np.ndarray) and csi_field.size > 0:
                        if csi_field.dtype == 'object':
                            actual_csi = csi_field.flat[0] if csi_field.size > 0 else None
                        else:
                            actual_csi = csi_field

                        if actual_csi is not None and isinstance(actual_csi, np.ndarray) and actual_csi.size > 10:
                            all_csi_data.append(actual_csi)

                            # Extraer etiqueta
                            if 'MPIs_label' in struct_data.dtype.names:
                                label_field = struct_data['MPIs_label']
                                if isinstance(label_field, np.ndarray):
                                    label_val = label_field.flat[0] if label_field.size > 0 else 0
                                else:
                                    label_val = label_field
                                all_labels.append(int(label_val))
                            else:
                                all_labels.append(i % 3)  # Etiqueta sintética

            except Exception as e:
                continue

        if len(all_csi_data) == 0:
            print("❌ No se extrajeron datos CSI de las estructuras")
            return

        print(f"📊 Extraídos {len(all_csi_data)} arrays CSI")

        # Verificar formas y apilar/concatenar
        shapes = [arr.shape for arr in all_csi_data[:50]]  # Verificar primeros 50
        from collections import Counter
        shape_counts = Counter(shapes)

        if len(shape_counts) == 1:
            # Todas iguales
            self.csi_data = np.stack(all_csi_data, axis=0)
            self.labels = np.array(all_labels)
        else:
            # Usar la forma más común
            most_common_shape = shape_counts.most_common(1)[0][0]
            filtered_csi = [arr for arr in all_csi_data if arr.shape == most_common_shape]
            filtered_labels = [all_labels[i] for i, arr in enumerate(all_csi_data) if arr.shape == most_common_shape]

            if len(filtered_csi) > 0:
                self.csi_data = np.stack(filtered_csi, axis=0)
                self.labels = np.array(filtered_labels)
            else:
                print("❌ No se pudo crear matriz CSI consistente")
                return

        print(f"✅ CSI extraído: {self.csi_data.shape}")
        print(f"✅ Labels extraídas: {self.labels.shape}")
        print(f"✅ Etiquetas únicas: {np.unique(self.labels)}")

        # Crear nombres de actividades
        unique_labels = np.unique(self.labels)
        self.activities = [f'Actividad_{int(label)}' for label in unique_labels]

        # FIX: Convertir datos 4D a 3D si es necesario
        if len(self.csi_data.shape) == 4:
            print(f"🔄 Convirtiendo datos 4D a 3D...")
            original_shape = self.csi_data.shape

            # (1000, 3, 3, 30) -> (3000, 3, 30)
            self.csi_data = self.csi_data.reshape(-1, self.csi_data.shape[2], self.csi_data.shape[3])

            # Ajustar etiquetas
            if self.labels is not None:
                repeated_labels = []
                for i, label in enumerate(self.labels):
                    repeated_labels.extend([label] * original_shape[1])
                self.labels = np.array(repeated_labels)

            print(f"✅ Conversión 4D->3D: {original_shape} -> {self.csi_data.shape}")

        print(f"✅ Actividades creadas: {self.activities}")

    def analyze_data_structure(self):
        """
        Analiza la estructura de los datos CSI
        """
        print("=== ANÁLISIS DE ESTRUCTURA DE DATOS ===\n")

        if self.raw_data is None:
            print("No hay datos cargados. Use load_wimir_data() primero.")
            return

        for key, value in self.raw_data.items():
            print(f"Clave: {key}")
            print(f"  Tipo: {type(value)}")
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")

            if hasattr(value, 'shape') and len(value.shape) > 0 and value.size > 0:
                try:
                    print(f"  Min: {np.min(value):.4f}")
                    print(f"  Max: {np.max(value):.4f}")
                    print(f"  Mean: {np.mean(value):.4f}")
                    print(f"  Std: {np.std(value):.4f}")
                except:
                    print("  No se pudieron calcular estadísticas")
            print()

    def plot_raw_csi_signals(self, n_samples=5, duration=2000, figsize=(15, 12), save_path=None):
        """
        Visualiza señales CSI en crudo

        Args:
            n_samples (int): Número de muestras a visualizar
            duration (int): Duración en muestras
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        if n_samples == 1:
            axes = [axes]

        for i in range(min(n_samples, len(axes))):
            # Seleccionar muestra
            if len(self.csi_data.shape) == 3:
                # Formato [tiempo, antenas, subportadoras]
                sample = self.csi_data[:duration, 0, 0]  # Primera antena, primera subportadora
            elif len(self.csi_data.shape) == 2:
                # Formato [tiempo, canales]
                sample = self.csi_data[:duration, i % self.csi_data.shape[1]]
            else:
                sample = self.csi_data[:duration]

            axes[i].plot(sample, linewidth=0.8, color='blue', alpha=0.7)
            axes[i].set_title(f'Señal CSI Raw - Muestra {i + 1}')
            axes[i].set_xlabel('Tiempo (muestras)')
            axes[i].set_ylabel('Amplitud')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def analyze_signal_spectrum(self, fs=1000, n_samples=3, figsize=(15, 12), save_path=None):
        """
        Analiza el espectro de frecuencias de las señales CSI

        Args:
            fs (int): Frecuencia de muestreo
            n_samples (int): Número de muestras a analizar
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(n_samples, 2, figsize=figsize)

        for i in range(n_samples):
            # Seleccionar señal
            if len(self.csi_data.shape) == 3:
                signal_data = self.csi_data[:2000, 0, i % self.csi_data.shape[2]]
            else:
                signal_data = self.csi_data[:2000, i % self.csi_data.shape[1]]

            # Dominio temporal
            axes[i, 0].plot(signal_data, color='blue', alpha=0.7)
            axes[i, 0].set_title(f'Señal en Tiempo - Canal {i + 1}')
            axes[i, 0].set_xlabel('Muestras')
            axes[i, 0].set_ylabel('Amplitud')
            axes[i, 0].grid(True, alpha=0.3)

            # Dominio frecuencial
            freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
            axes[i, 1].semilogy(freqs, psd, color='red', alpha=0.7)
            axes[i, 1].set_title(f'PSD - Canal {i + 1}')
            axes[i, 1].set_xlabel('Frecuencia (Hz)')
            axes[i, 1].set_ylabel('PSD')
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def create_spectrograms(self, fs=1000, n_samples=3, figsize=(15, 6), save_path=None):
        """
        Crea espectrogramas de las señales CSI

        Args:
            fs (int): Frecuencia de muestreo
            n_samples (int): Número de espectrogramas
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = [axes]

        for i in range(n_samples):
            # Seleccionar señal
            if len(self.csi_data.shape) == 3:
                signal_data = self.csi_data[:3000, 0, i]
            else:
                signal_data = self.csi_data[:3000, i % self.csi_data.shape[1]]

            # Crear espectrograma
            f, t, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=128, noverlap=64)

            # Visualizar
            im = axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            axes[i].set_title(f'Espectrograma - Canal {i + 1}')
            axes[i].set_xlabel('Tiempo (s)')
            axes[i].set_ylabel('Frecuencia (Hz)')
            plt.colorbar(im, ax=axes[i], label='Potencia (dB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def analyze_activity_distribution(self, activity_names=None, figsize=(12, 6), save_path=None):
        """
        Analiza la distribución de actividades en el dataset

        Args:
            activity_names (list): Nombres de las actividades
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.labels is None:
            print("No hay etiquetas cargadas")
            return

        # Contar actividades
        activity_counts = Counter(self.labels.flatten() if hasattr(self.labels, 'flatten') else self.labels)

        # Crear nombres si no se proporcionan
        if activity_names is None:
            activity_names = self.activities or [f'Actividad {int(i)}' for i in sorted(activity_counts.keys())]

        # Gráfico de barras y pastel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        colors = sns.color_palette("husl", len(activities))

        # Gráfico de barras
        bars = ax1.bar(range(len(activities)), counts, color=colors)
        ax1.set_xticks(range(len(activities)))
        ax1.set_xticklabels([activity_names[int(i)] if i < len(activity_names) else f'Act_{int(i)}'
                             for i in activities], rotation=45, ha='right')
        ax1.set_title('Distribución de Actividades')
        ax1.set_ylabel('Número de Muestras')
        ax1.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     str(int(count)), ha='center', va='bottom')

        # Gráfico de pastel
        ax2.pie(counts, labels=[activity_names[int(i)] if i < len(activity_names) else f'Act_{int(i)}'
                                for i in activities],
                autopct='%1.1f%%', colors=colors)
        ax2.set_title('Proporción de Actividades')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

        # Estadísticas
        print("=== ESTADÍSTICAS DE ACTIVIDADES ===")
        total_samples = sum(counts)
        for activity, count in activity_counts.items():
            activity_name = (activity_names[int(activity)] if activity < len(activity_names)
                             else f"Actividad {int(activity)}")
            percentage = (count / total_samples) * 100
            print(f"{activity_name}: {int(count)} muestras ({percentage:.1f}%)")

    def analyze_signal_quality(self, figsize=(15, 10), save_path=None):
        """
        Analiza la calidad de las señales CSI

        Args:
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        print("=== ANÁLISIS DE CALIDAD DE SEÑAL ===")

        # Estadísticas básicas
        print(f"Dimensiones: {self.csi_data.shape}")

        if len(self.csi_data.shape) == 3:
            print(f"  Tiempo: {self.csi_data.shape[0]} muestras")
            print(f"  Antenas: {self.csi_data.shape[1]}")
            print(f"  Subportadoras: {self.csi_data.shape[2]}")

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # SNR aproximado por antena
            snr_by_antenna = []
            n_antennas_to_analyze = min(4, self.csi_data.shape[1])

            for ant in range(n_antennas_to_analyze):
                signal_power = np.mean(self.csi_data[:, ant, :] ** 2)
                noise_estimate = np.var(self.csi_data[:, ant, :] -
                                        signal.medfilt(self.csi_data[:, ant, :], kernel_size=3, axis=0))
                snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                snr_by_antenna.append(snr)

            # SNR por antena
            axes[0, 0].bar(range(len(snr_by_antenna)), snr_by_antenna, color='skyblue')
            axes[0, 0].set_title('SNR Aproximado por Antena')
            axes[0, 0].set_xlabel('Antena')
            axes[0, 0].set_ylabel('SNR (dB)')
            axes[0, 0].grid(True, alpha=0.3)

            # Variabilidad de la señal
            signal_std = np.std(self.csi_data, axis=0)
            im1 = axes[0, 1].imshow(signal_std, aspect='auto', cmap='viridis')
            axes[0, 1].set_title('Desviación Estándar por Antena-Subportadora')
            axes[0, 1].set_xlabel('Subportadora')
            axes[0, 1].set_ylabel('Antena')
            plt.colorbar(im1, ax=axes[0, 1])

            # Correlación entre antenas
            if self.csi_data.shape[1] > 1:
                antenna_signals = self.csi_data[:1000, :, 0].T  # Primera subportadora
                correlation_matrix = np.corrcoef(antenna_signals)

                im2 = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 0].set_title('Correlación entre Antenas')
                axes[1, 0].set_xlabel('Antena')
                axes[1, 0].set_ylabel('Antena')
                plt.colorbar(im2, ax=axes[1, 0])

            # Histograma de amplitudes
            sample_data = self.csi_data[:, 0, 0].flatten()
            axes[1, 1].hist(sample_data, bins=50, alpha=0.7, density=True, color='green')
            axes[1, 1].set_title('Distribución de Amplitudes')
            axes[1, 1].set_xlabel('Amplitud')
            axes[1, 1].set_ylabel('Densidad')
            axes[1, 1].grid(True, alpha=0.3)

        else:
            print(f"  Tiempo: {self.csi_data.shape[0]} muestras")
            print(f"  Canales: {self.csi_data.shape[1]}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def detect_outliers(self, threshold=3, figsize=(12, 4), save_path=None):
        """
        Detecta outliers en las señales CSI usando Z-score

        Args:
            threshold (float): Umbral de Z-score
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return None

        print("=== DETECCIÓN DE OUTLIERS ===")

        # Calcular Z-scores
        if len(self.csi_data.shape) == 3:
            reshaped_data = self.csi_data.reshape(self.csi_data.shape[0], -1)
        else:
            reshaped_data = self.csi_data

        z_scores = np.abs((reshaped_data - np.mean(reshaped_data, axis=0)) /
                          (np.std(reshaped_data, axis=0) + 1e-10))
        outliers = z_scores > threshold

        outlier_percentage = (np.sum(outliers) / outliers.size) * 100
        print(f"Porcentaje de outliers (Z-score > {threshold}): {outlier_percentage:.2f}%")

        # Visualizar outliers
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Distribución de Z-scores
        ax1.hist(z_scores.flatten(), bins=100, alpha=0.7, density=True, color='blue')
        ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        ax1.set_title('Distribución de Z-scores')
        ax1.set_xlabel('Z-score')
        ax1.set_ylabel('Densidad')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Outliers por muestra temporal
        outlier_counts = np.sum(outliers, axis=1)
        ax2.plot(outlier_counts[:1000], color='red', alpha=0.7)
        ax2.set_title('Outliers por Muestra Temporal')
        ax2.set_xlabel('Tiempo (muestras)')
        ax2.set_ylabel('Número de Outliers')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

        return outliers

    def generate_data_summary_report(self):
        """
        Genera un reporte resumen de los datos cargados
        """
        print("\n" + "=" * 60)
        print("REPORTE RESUMEN DE DATOS")
        print("=" * 60)

        if self.csi_data is not None:
            print(f"📊 DATOS CSI:")
            print(f"  Shape: {self.csi_data.shape}")
            print(f"  Tipo: {self.csi_data.dtype}")
            print(f"  Tamaño total: {self.csi_data.size:,} elementos")
            print(f"  Memoria: {self.csi_data.nbytes / 1024 ** 2:.2f} MB")

            if len(self.csi_data.shape) == 3:
                print(
                    f"  Estructura: [tiempo={self.csi_data.shape[0]}, antenas={self.csi_data.shape[1]}, subportadoras={self.csi_data.shape[2]}]")

            print(f"  Rango: [{np.min(self.csi_data):.4f}, {np.max(self.csi_data):.4f}]")
            print(f"  Media: {np.mean(self.csi_data):.4f}")
            print(f"  Desv. estándar: {np.std(self.csi_data):.4f}")

        if self.labels is not None:
            print(f"\n🏷️  ETIQUETAS:")
            print(f"  Shape: {self.labels.shape}")
            print(f"  Etiquetas únicas: {len(np.unique(self.labels))}")
            print(f"  Rango: {np.min(self.labels)} - {np.max(self.labels)}")

            if self.activities:
                print(f"  Actividades: {', '.join(self.activities)}")

        if self.raw_data:
            print(f"\n📁 DATOS RAW:")
            print(f"  Claves totales: {len(self.raw_data)}")
            print(f"  Claves: {list(self.raw_data.keys())}")

        print("\n" + "=" * 60)

    def export_processed_data(self, output_path, format='npz'):
        """
        Exporta los datos procesados

        Args:
            output_path (str): Ruta de salida
            format (str): Formato ('npz', 'mat', 'csv')
        """
        if self.csi_data is None:
            print("No hay datos para exportar")
            return

        print(f"Exportando datos en formato {format}...")

        if format == 'npz':
            if self.labels is not None:
                np.savez(output_path, csi_data=self.csi_data, labels=self.labels,
                         activities=self.activities)
            else:
                np.savez(output_path, csi_data=self.csi_data)

        elif format == 'mat':
            from scipy.io import savemat
            data_dict = {'csi_data': self.csi_data}
            if self.labels is not None:
                data_dict['labels'] = self.labels
            savemat(output_path, data_dict)

        elif format == 'csv':
            # Para CSV, aplanar los datos
            if len(self.csi_data.shape) == 3:
                flattened_data = self.csi_data.reshape(self.csi_data.shape[0], -1)
            else:
                flattened_data = self.csi_data

            df = pd.DataFrame(flattened_data)
            if self.labels is not None:
                df['label'] = self.labels[:len(df)]  # Ajustar longitud si es necesario

            df.to_csv(output_path, index=False)

        print(f"Datos exportados a: {output_path}")


def demo_data_loading():
    """
    Demostración del cargador de datos con datos sintéticos
    """
    print("=== DEMOSTRACIÓN DEL CARGADOR DE DATOS ===")

    # Crear datos sintéticos para la demostración
    print("Creando datos sintéticos para demostración...")

    np.random.seed(42)
    n_samples = 5000
    n_antennas = 4
    n_subcarriers = 56

    # Simular datos CSI
    csi_data = np.zeros((n_samples, n_antennas, n_subcarriers))
    activities = ['caminar', 'correr', 'sentado', 'parado', 'acostado']
    labels = np.zeros(n_samples // 100)  # Una etiqueta cada 100 muestras

    for i in range(n_samples):
        activity_idx = (i // 1000) % len(activities)

        # Patrones diferentes por actividad
        if activity_idx == 0:  # caminar
            pattern = np.sin(2 * np.pi * 0.02 * i) + 0.5 * np.sin(2 * np.pi * 0.05 * i)
        elif activity_idx == 1:  # correr
            pattern = 2 * np.sin(2 * np.pi * 0.04 * i) + 0.3 * np.random.randn()
        elif activity_idx == 2:  # sentado
            pattern = 0.1 * np.sin(2 * np.pi * 0.001 * i) + 0.05 * np.random.randn()
        elif activity_idx == 3:  # parado
            pattern = 0.05 * np.sin(2 * np.pi * 0.002 * i) + 0.02 * np.random.randn()
        else:  # acostado
            pattern = 0.02 * np.random.randn()

        # Aplicar a todas las antenas y subportadoras
        for ant in range(n_antennas):
            for sub in range(n_subcarriers):
                variation = 0.1 * np.random.randn()
                csi_data[i, ant, sub] = pattern + variation

        # Etiquetas
        if i % 100 == 0:
            labels[i // 100] = activity_idx

    # Crear cargador y asignar datos sintéticos
    loader = WiFiDataLoader()
    loader.csi_data = csi_data
    loader.labels = labels
    loader.activities = activities

    print(f"Datos sintéticos creados: {csi_data.shape}")

    # Demostrar funcionalidades
    loader.analyze_data_structure()
    loader.plot_raw_csi_signals(n_samples=3)
    loader.analyze_signal_spectrum(n_samples=3)
    loader.create_spectrograms(n_samples=3)
    loader.analyze_activity_distribution()
    loader.analyze_signal_quality()
    loader.detect_outliers()
    loader.generate_data_summary_report()

    return loader


if __name__ == "__main__":
    # Ejecutar demostración
    loader = demo_data_loading()
    print("\n✅ Demostración del cargador de datos completada!")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy import signal
import h5py
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class WiFiDataLoader:
    """
    Cargador y explorador de datos para el dataset WI-MIR
    """

    def __init__(self, dataset_path=None):
        """
        Inicializa el cargador de datos

        Args:
            dataset_path (str): Ruta al archivo del dataset
        """
        self.dataset_path = dataset_path
        self.raw_data = None
        self.csi_data = None
        self.labels = None
        self.activities = None
        self.metadata = {}

    def load_wimir_data(self, file_path=None):
        """
        Carga datos del dataset WI-MIR desde archivo .mat

        Args:
            file_path (str): Ruta al archivo (opcional)

        Returns:
            dict: Diccionario con datos CSI y metadatos
        """
        if file_path is None:
            file_path = self.dataset_path

        if file_path is None:
            raise ValueError("Se debe proporcionar una ruta al archivo")

        try:
            # Cargar archivo .mat
            print(f"Cargando datos desde: {file_path}")
            data = loadmat(file_path)

            # Filtrar claves privadas de MATLAB
            clean_data = {k: v for k, v in data.items() if not k.startswith('__')}

            self.raw_data = clean_data

            print(f"Archivo cargado exitosamente")
            print(f"Claves disponibles: {list(clean_data.keys())}")

            # Extraer datos CSI y etiquetas automáticamente
            self._extract_csi_and_labels()

            return clean_data

        except Exception as e:
            print(f"Error al cargar archivo: {e}")
            raise

    def _extract_csi_and_labels(self):
        """
        Extrae automáticamente datos CSI y etiquetas del dataset cargado
        VERSIÓN CORREGIDA PARA ESTRUCTURAS MATLAB
        """
        import numpy as np
        from collections import Counter

        if self.raw_data is None:
            print("❌ No hay datos raw cargados")
            return

        print("🔍 Extrayendo datos de estructuras MATLAB...")

        if 'Raw_Cell_Matrix' not in self.raw_data:
            print("❌ No se encontró Raw_Cell_Matrix")
            return

        cell_matrix = self.raw_data['Raw_Cell_Matrix']
        print(f"📦 Cell matrix shape: {cell_matrix.shape}")

        all_csi_data = []
        all_labels = []

        print("🔄 Procesando estructuras...")

        # Procesar estructuras
        processed_count = 0
        for i in range(min(1000, cell_matrix.size)):
            try:
                # Extraer estructura
                if cell_matrix.ndim == 2:
                    struct = cell_matrix[i, 0]
                else:
                    struct = cell_matrix.flat[i]

                if not isinstance(struct, np.ndarray) or struct.size == 0:
                    continue

                # La estructura debería tener un solo elemento
                struct_data = struct[0, 0] if struct.ndim == 2 else struct.flat[0]

                # Verificar que tiene los campos necesarios
                if hasattr(struct_data,
                           'dtype') and struct_data.dtype.names is not None and 'csi' in struct_data.dtype.names:
                    csi_field = struct_data['csi']

                    # Extraer datos CSI reales
                    if isinstance(csi_field, np.ndarray) and csi_field.size > 0:
                        if csi_field.dtype == 'object' and csi_field.size > 0:
                            actual_csi = csi_field.flat[0] if csi_field.size > 0 else None
                        else:
                            actual_csi = csi_field

                        if actual_csi is not None and isinstance(actual_csi, np.ndarray) and actual_csi.size > 10:
                            all_csi_data.append(actual_csi)

                            # Extraer etiqueta
                            if 'MPIs_label' in struct_data.dtype.names:
                                label_field = struct_data['MPIs_label']
                                if isinstance(label_field, np.ndarray) and label_field.size > 0:
                                    label_val = label_field.flat[0]
                                else:
                                    label_val = label_field if np.isscalar(label_field) else 0

                                try:
                                    all_labels.append(int(float(label_val)))
                                except:
                                    all_labels.append(processed_count % 3)
                            else:
                                all_labels.append(processed_count % 3)

                            processed_count += 1

                            if processed_count % 100 == 0:
                                print(f"   Procesadas {processed_count} estructuras...")

            except Exception as e:
                continue

        print(f"📊 Procesamiento completado: {processed_count} estructuras procesadas")

        if len(all_csi_data) == 0:
            print("❌ No se extrajeron datos CSI de las estructuras")
            return

        print(f"✅ Extraídos {len(all_csi_data)} arrays CSI")

        # Verificar formas y crear matriz final
        if len(all_csi_data) > 0:
            # Analizar formas
            shapes = [arr.shape for arr in all_csi_data[:50]]
            shape_counts = Counter(shapes)

            print(f"📐 Formas encontradas: {dict(shape_counts)}")

            if len(shape_counts) == 1:
                self.csi_data = np.stack(all_csi_data, axis=0)
                self.labels = np.array(all_labels)
            else:
                most_common_shape = shape_counts.most_common(1)[0][0]
                filtered_csi = [arr for arr in all_csi_data if arr.shape == most_common_shape]
                filtered_labels = [all_labels[i] for i, arr in enumerate(all_csi_data)
                                   if arr.shape == most_common_shape]

                if len(filtered_csi) > 0:
                    self.csi_data = np.stack(filtered_csi, axis=0)
                    self.labels = np.array(filtered_labels)

            print(f"✅ CSI final shape: {self.csi_data.shape}")
            print(f"✅ Labels final shape: {self.labels.shape}")

            # Crear nombres de actividades
            unique_labels = np.unique(self.labels)
            self.activities = [f'Actividad_{int(label)}' for label in unique_labels]

            # FIX 4D->3D: AÑADIR ESTAS LÍNEAS AQUÍ ↓
            if len(self.csi_data.shape) == 4:
                print(f"🔄 Convirtiendo datos 4D a 3D...")
                original_shape = self.csi_data.shape

                # (1000, 3, 3, 30) -> (3000, 3, 30)
                self.csi_data = self.csi_data.reshape(-1, self.csi_data.shape[2], self.csi_data.shape[3])

                # Ajustar etiquetas
                if self.labels is not None:
                    repeated_labels = []
                    for i, label in enumerate(self.labels):
                        repeated_labels.extend([label] * original_shape[1])
                    self.labels = np.array(repeated_labels)

                print(f"✅ Conversión 4D->3D: {original_shape} -> {self.csi_data.shape}")

            print(f"✅ Actividades creadas: {self.activities}")

    def analyze_data_structure(self):
        """
        Analiza la estructura de los datos CSI
        """
        print("=== ANÁLISIS DE ESTRUCTURA DE DATOS ===\n")

        if self.raw_data is None:
            print("No hay datos cargados. Use load_wimir_data() primero.")
            return

        for key, value in self.raw_data.items():
            print(f"Clave: {key}")
            print(f"  Tipo: {type(value)}")
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")

            if hasattr(value, 'shape') and len(value.shape) > 0 and value.size > 0:
                try:
                    print(f"  Min: {np.min(value):.4f}")
                    print(f"  Max: {np.max(value):.4f}")
                    print(f"  Mean: {np.mean(value):.4f}")
                    print(f"  Std: {np.std(value):.4f}")
                except:
                    print("  No se pudieron calcular estadísticas")
            print()

    def plot_raw_csi_signals(self, n_samples=5, duration=2000, figsize=(15, 12), save_path=None):
        """
        Visualiza señales CSI en crudo

        Args:
            n_samples (int): Número de muestras a visualizar
            duration (int): Duración en muestras
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(n_samples, 1, figsize=figsize)
        if n_samples == 1:
            axes = [axes]

        for i in range(min(n_samples, len(axes))):
            # Seleccionar muestra
            if len(self.csi_data.shape) == 3:
                # Formato [tiempo, antenas, subportadoras]
                sample = self.csi_data[:duration, 0, 0]  # Primera antena, primera subportadora
            elif len(self.csi_data.shape) == 2:
                # Formato [tiempo, canales]
                sample = self.csi_data[:duration, i % self.csi_data.shape[1]]
            else:
                sample = self.csi_data[:duration]

            axes[i].plot(sample, linewidth=0.8, color='blue', alpha=0.7)
            axes[i].set_title(f'Señal CSI Raw - Muestra {i+1}')
            axes[i].set_xlabel('Tiempo (muestras)')
            axes[i].set_ylabel('Amplitud')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def analyze_signal_spectrum(self, fs=1000, n_samples=3, figsize=(15, 12), save_path=None):
        """
        Analiza el espectro de frecuencias de las señales CSI

        Args:
            fs (int): Frecuencia de muestreo
            n_samples (int): Número de muestras a analizar
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(n_samples, 2, figsize=figsize)

        for i in range(n_samples):
            # Seleccionar señal
            if len(self.csi_data.shape) == 3:
                signal_data = self.csi_data[:2000, 0, i % self.csi_data.shape[2]]
            else:
                signal_data = self.csi_data[:2000, i % self.csi_data.shape[1]]

            # Dominio temporal
            axes[i, 0].plot(signal_data, color='blue', alpha=0.7)
            axes[i, 0].set_title(f'Señal en Tiempo - Canal {i+1}')
            axes[i, 0].set_xlabel('Muestras')
            axes[i, 0].set_ylabel('Amplitud')
            axes[i, 0].grid(True, alpha=0.3)

            # Dominio frecuencial
            freqs, psd = signal.welch(signal_data, fs=fs, nperseg=256)
            axes[i, 1].semilogy(freqs, psd, color='red', alpha=0.7)
            axes[i, 1].set_title(f'PSD - Canal {i+1}')
            axes[i, 1].set_xlabel('Frecuencia (Hz)')
            axes[i, 1].set_ylabel('PSD')
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def create_spectrograms(self, fs=1000, n_samples=3, figsize=(15, 6), save_path=None):
        """
        Crea espectrogramas de las señales CSI

        Args:
            fs (int): Frecuencia de muestreo
            n_samples (int): Número de espectrogramas
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        fig, axes = plt.subplots(1, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = [axes]

        for i in range(n_samples):
            # Seleccionar señal
            if len(self.csi_data.shape) == 3:
                signal_data = self.csi_data[:3000, 0, i]
            else:
                signal_data = self.csi_data[:3000, i % self.csi_data.shape[1]]

            # Crear espectrograma
            f, t, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=128, noverlap=64)

            # Visualizar
            im = axes[i].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            axes[i].set_title(f'Espectrograma - Canal {i+1}')
            axes[i].set_xlabel('Tiempo (s)')
            axes[i].set_ylabel('Frecuencia (Hz)')
            plt.colorbar(im, ax=axes[i], label='Potencia (dB)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def analyze_activity_distribution(self, activity_names=None, figsize=(12, 6), save_path=None):
        """
        Analiza la distribución de actividades en el dataset

        Args:
            activity_names (list): Nombres de las actividades
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.labels is None:
            print("No hay etiquetas cargadas")
            return

        # Contar actividades
        activity_counts = Counter(self.labels.flatten() if hasattr(self.labels, 'flatten') else self.labels)

        # Crear nombres si no se proporcionan
        if activity_names is None:
            activity_names = self.activities or [f'Actividad {int(i)}' for i in sorted(activity_counts.keys())]

        # Gráfico de barras y pastel
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        colors = sns.color_palette("husl", len(activities))

        # Gráfico de barras
        bars = ax1.bar(range(len(activities)), counts, color=colors)
        ax1.set_xticks(range(len(activities)))
        ax1.set_xticklabels([activity_names[int(i)] if i < len(activity_names) else f'Act_{int(i)}'
                            for i in activities], rotation=45, ha='right')
        ax1.set_title('Distribución de Actividades')
        ax1.set_ylabel('Número de Muestras')
        ax1.grid(True, alpha=0.3)

        # Añadir valores en las barras
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(int(count)), ha='center', va='bottom')

        # Gráfico de pastel
        ax2.pie(counts, labels=[activity_names[int(i)] if i < len(activity_names) else f'Act_{int(i)}'
                               for i in activities],
                autopct='%1.1f%%', colors=colors)
        ax2.set_title('Proporción de Actividades')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

        # Estadísticas
        print("=== ESTADÍSTICAS DE ACTIVIDADES ===")
        total_samples = sum(counts)
        for activity, count in activity_counts.items():
            activity_name = (activity_names[int(activity)] if activity < len(activity_names)
                           else f"Actividad {int(activity)}")
            percentage = (count / total_samples) * 100
            print(f"{activity_name}: {int(count)} muestras ({percentage:.1f}%)")

    def analyze_signal_quality(self, figsize=(15, 10), save_path=None):
        """
        Analiza la calidad de las señales CSI

        Args:
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return

        print("=== ANÁLISIS DE CALIDAD DE SEÑAL ===")

        # Estadísticas básicas
        print(f"Dimensiones: {self.csi_data.shape}")

        if len(self.csi_data.shape) == 3:
            print(f"  Tiempo: {self.csi_data.shape[0]} muestras")
            print(f"  Antenas: {self.csi_data.shape[1]}")
            print(f"  Subportadoras: {self.csi_data.shape[2]}")

            fig, axes = plt.subplots(2, 2, figsize=figsize)

            # SNR aproximado por antena
            snr_by_antenna = []
            n_antennas_to_analyze = min(4, self.csi_data.shape[1])

            for ant in range(n_antennas_to_analyze):
                signal_power = np.mean(self.csi_data[:, ant, :] ** 2)
                noise_estimate = np.var(self.csi_data[:, ant, :] -
                                      signal.medfilt(self.csi_data[:, ant, :], kernel_size=3, axis=0))
                snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                snr_by_antenna.append(snr)

            # SNR por antena
            axes[0, 0].bar(range(len(snr_by_antenna)), snr_by_antenna, color='skyblue')
            axes[0, 0].set_title('SNR Aproximado por Antena')
            axes[0, 0].set_xlabel('Antena')
            axes[0, 0].set_ylabel('SNR (dB)')
            axes[0, 0].grid(True, alpha=0.3)

            # Variabilidad de la señal
            signal_std = np.std(self.csi_data, axis=0)
            im1 = axes[0, 1].imshow(signal_std, aspect='auto', cmap='viridis')
            axes[0, 1].set_title('Desviación Estándar por Antena-Subportadora')
            axes[0, 1].set_xlabel('Subportadora')
            axes[0, 1].set_ylabel('Antena')
            plt.colorbar(im1, ax=axes[0, 1])

            # Correlación entre antenas
            if self.csi_data.shape[1] > 1:
                antenna_signals = self.csi_data[:1000, :, 0].T  # Primera subportadora
                correlation_matrix = np.corrcoef(antenna_signals)

                im2 = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 0].set_title('Correlación entre Antenas')
                axes[1, 0].set_xlabel('Antena')
                axes[1, 0].set_ylabel('Antena')
                plt.colorbar(im2, ax=axes[1, 0])

            # Histograma de amplitudes
            sample_data = self.csi_data[:, 0, 0].flatten()
            axes[1, 1].hist(sample_data, bins=50, alpha=0.7, density=True, color='green')
            axes[1, 1].set_title('Distribución de Amplitudes')
            axes[1, 1].set_xlabel('Amplitud')
            axes[1, 1].set_ylabel('Densidad')
            axes[1, 1].grid(True, alpha=0.3)

        else:
            print(f"  Tiempo: {self.csi_data.shape[0]} muestras")
            print(f"  Canales: {self.csi_data.shape[1]}")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

    def detect_outliers(self, threshold=3, figsize=(12, 4), save_path=None):
        """
        Detecta outliers en las señales CSI usando Z-score

        Args:
            threshold (float): Umbral de Z-score
            figsize (tuple): Tamaño de la figura
            save_path (str): Ruta para guardar la figura
        """
        if self.csi_data is None:
            print("No hay datos CSI cargados")
            return None

        print("=== DETECCIÓN DE OUTLIERS ===")

        # Calcular Z-scores
        if len(self.csi_data.shape) == 3:
            reshaped_data = self.csi_data.reshape(self.csi_data.shape[0], -1)
        else:
            reshaped_data = self.csi_data

        z_scores = np.abs((reshaped_data - np.mean(reshaped_data, axis=0)) /
                         (np.std(reshaped_data, axis=0) + 1e-10))
        outliers = z_scores > threshold

        outlier_percentage = (np.sum(outliers) / outliers.size) * 100
        print(f"Porcentaje de outliers (Z-score > {threshold}): {outlier_percentage:.2f}%")

        # Visualizar outliers
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Distribución de Z-scores
        ax1.hist(z_scores.flatten(), bins=100, alpha=0.7, density=True, color='blue')
        ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        ax1.set_title('Distribución de Z-scores')
        ax1.set_xlabel('Z-score')
        ax1.set_ylabel('Densidad')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Outliers por muestra temporal
        outlier_counts = np.sum(outliers, axis=1)
        ax2.plot(outlier_counts[:1000], color='red', alpha=0.7)
        ax2.set_title('Outliers por Muestra Temporal')
        ax2.set_xlabel('Tiempo (muestras)')
        ax2.set_ylabel('Número de Outliers')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura guardada en: {save_path}")

        plt.show()

        return outliers

    def generate_data_summary_report(self):
        """
        Genera un reporte resumen de los datos cargados
        """
        print("\n" + "="*60)
        print("REPORTE RESUMEN DE DATOS")
        print("="*60)

        if self.csi_data is not None:
            print(f"📊 DATOS CSI:")
            print(f"  Shape: {self.csi_data.shape}")
            print(f"  Tipo: {self.csi_data.dtype}")
            print(f"  Tamaño total: {self.csi_data.size:,} elementos")
            print(f"  Memoria: {self.csi_data.nbytes / 1024**2:.2f} MB")

            if len(self.csi_data.shape) == 3:
                print(f"  Estructura: [tiempo={self.csi_data.shape[0]}, antenas={self.csi_data.shape[1]}, subportadoras={self.csi_data.shape[2]}]")

            print(f"  Rango: [{np.min(self.csi_data):.4f}, {np.max(self.csi_data):.4f}]")
            print(f"  Media: {np.mean(self.csi_data):.4f}")
            print(f"  Desv. estándar: {np.std(self.csi_data):.4f}")

        if self.labels is not None:
            print(f"\n🏷️  ETIQUETAS:")
            print(f"  Shape: {self.labels.shape}")
            print(f"  Etiquetas únicas: {len(np.unique(self.labels))}")
            print(f"  Rango: {np.min(self.labels)} - {np.max(self.labels)}")

            if self.activities:
                print(f"  Actividades: {', '.join(self.activities)}")

        if self.raw_data:
            print(f"\n📁 DATOS RAW:")
            print(f"  Claves totales: {len(self.raw_data)}")
            print(f"  Claves: {list(self.raw_data.keys())}")

        print("\n" + "="*60)

    def export_processed_data(self, output_path, format='npz'):
        """
        Exporta los datos procesados

        Args:
            output_path (str): Ruta de salida
            format (str): Formato ('npz', 'mat', 'csv')
        """
        if self.csi_data is None:
            print("No hay datos para exportar")
            return

        print(f"Exportando datos en formato {format}...")

        if format == 'npz':
            if self.labels is not None:
                np.savez(output_path, csi_data=self.csi_data, labels=self.labels,
                        activities=self.activities)
            else:
                np.savez(output_path, csi_data=self.csi_data)

        elif format == 'mat':
            from scipy.io import savemat
            data_dict = {'csi_data': self.csi_data}
            if self.labels is not None:
                data_dict['labels'] = self.labels
            savemat(output_path, data_dict)

        elif format == 'csv':
            # Para CSV, aplanar los datos
            if len(self.csi_data.shape) == 3:
                flattened_data = self.csi_data.reshape(self.csi_data.shape[0], -1)
            else:
                flattened_data = self.csi_data

            df = pd.DataFrame(flattened_data)
            if self.labels is not None:
                df['label'] = self.labels[:len(df)]  # Ajustar longitud si es necesario

            df.to_csv(output_path, index=False)

        print(f"Datos exportados a: {output_path}")


def demo_data_loading():
    """
    Demostración del cargador de datos con datos sintéticos
    """
    print("=== DEMOSTRACIÓN DEL CARGADOR DE DATOS ===")

    # Crear datos sintéticos para la demostración
    print("Creando datos sintéticos para demostración...")

    np.random.seed(42)
    n_samples = 5000
    n_antennas = 4
    n_subcarriers = 56

    # Simular datos CSI
    csi_data = np.zeros((n_samples, n_antennas, n_subcarriers))
    activities = ['caminar', 'correr', 'sentado', 'parado', 'acostado']
    labels = np.zeros(n_samples // 100)  # Una etiqueta cada 100 muestras

    for i in range(n_samples):
        activity_idx = (i // 1000) % len(activities)

        # Patrones diferentes por actividad
        if activity_idx == 0:  # caminar
            pattern = np.sin(2 * np.pi * 0.02 * i) + 0.5 * np.sin(2 * np.pi * 0.05 * i)
        elif activity_idx == 1:  # correr
            pattern = 2 * np.sin(2 * np.pi * 0.04 * i) + 0.3 * np.random.randn()
        elif activity_idx == 2:  # sentado
            pattern = 0.1 * np.sin(2 * np.pi * 0.001 * i) + 0.05 * np.random.randn()
        elif activity_idx == 3:  # parado
            pattern = 0.05 * np.sin(2 * np.pi * 0.002 * i) + 0.02 * np.random.randn()
        else:  # acostado
            pattern = 0.02 * np.random.randn()

        # Aplicar a todas las antenas y subportadoras
        for ant in range(n_antennas):
            for sub in range(n_subcarriers):
                variation = 0.1 * np.random.randn()
                csi_data[i, ant, sub] = pattern + variation

        # Etiquetas
        if i % 100 == 0:
            labels[i // 100] = activity_idx

    # Crear cargador y asignar datos sintéticos
    loader = WiFiDataLoader()
    loader.csi_data = csi_data
    loader.labels = labels
    loader.activities = activities

    print(f"Datos sintéticos creados: {csi_data.shape}")

    # Demostrar funcionalidades
    loader.analyze_data_structure()
    loader.plot_raw_csi_signals(n_samples=3)
    loader.analyze_signal_spectrum(n_samples=3)
    loader.create_spectrograms(n_samples=3)
    loader.analyze_activity_distribution()
    loader.analyze_signal_quality()
    loader.detect_outliers()
    loader.generate_data_summary_report()

    return loader


if __name__ == "__main__":
    # Ejecutar demostración
    loader = demo_data_loading()
    print("\n✅ Demostración del cargador de datos completada!")