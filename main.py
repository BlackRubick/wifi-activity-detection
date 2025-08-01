
import argparse
import sys
import os
from pathlib import Path
import time
from datetime import datetime
import traceback

import numpy as np

# Agregar src al path para importaciones
sys.path.append(str(Path(__file__).parent / 'src'))

# Importar módulos del proyecto
try:
    from data_loader import WiFiDataLoader, demo_data_loading
    from preprocessing import CSIPreprocessingPipeline
    from feature_extraction import CSIFeatureExtractor
    from models import ActivityRecognitionModels
    from utils import (
        setup_logging,
        load_config,
        save_config,
        get_default_config,
        create_project_structure,
        set_random_seeds,
        ExperimentTracker,
        timer_decorator,
        format_duration,
        save_pickle
    )

    print("✓ Todos los módulos importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos: {e}")
    print("Asegúrate de que todos los archivos estén en src/")
    sys.exit(1)


class WiFiActivityDetectionPipeline:
    """
    Pipeline principal del proyecto
    """

    def __init__(self, config_path=None, data_path=None, output_path="results"):
        """
        Inicializa el pipeline
        """
        print("🚀 Inicializando pipeline...")

        # Cargar configuración
        if config_path and Path(config_path).exists():
            self.config = load_config(config_path)
            print(f"✓ Configuración cargada desde: {config_path}")
        else:
            self.config = get_default_config()
            if config_path:
                print(f"⚠️ Archivo {config_path} no encontrado. Usando configuración por defecto.")
            else:
                print("✓ Usando configuración por defecto")

        # Configurar rutas
        self.data_path = data_path
        self.output_path = Path(output_path)

        # Crear estructura de proyecto
        create_project_structure(self.output_path)

        # Configurar logging
        log_file = self.output_path / "logs" / f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logging(log_file=str(log_file))

        # Configurar reproducibilidad
        set_random_seeds(self.config['modeling']['random_state'])

        # Inicializar tracker de experimento
        self.tracker = ExperimentTracker(
            experiment_name="wifi_activity_detection",
            base_path=str(self.output_path / "experiments")
        )

        # Componentes del pipeline
        self.data_loader = None
        self.preprocessor = None
        self.feature_extractor = None
        self.models = None

        # Datos
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.labels = None

        self.logger.info("Pipeline inicializado correctamente")
        print("✓ Pipeline inicializado")

    @timer_decorator
    def load_data(self):
        """
        Paso 1: Carga y exploración de datos
        """
        print("\n" + "=" * 60)
        print("PASO 1: CARGA Y EXPLORACIÓN DE DATOS")
        print("=" * 60)

        if not self.data_path or not Path(self.data_path).exists():
            if self.data_path:
                print(f"⚠️ Archivo {self.data_path} no encontrado.")
            print("📊 Generando datos sintéticos para demostración...")
            self._create_synthetic_data()
            return

        # Cargar datos reales
        print(f"📁 Cargando datos desde: {self.data_path}")
        self.data_loader = WiFiDataLoader(self.data_path)

        try:
            self.data_loader.load_wimir_data()
            self.raw_data = self.data_loader.csi_data
            self.labels = self.data_loader.labels

            print("📊 Realizando exploración de datos...")
            self.data_loader.analyze_data_structure()

            # Crear visualizaciones CON FIX
            try:
                figures_path = self.output_path / "figures"
                figures_path.mkdir(exist_ok=True)  # ← AÑADIR ESTA LÍNEA

                self.data_loader.plot_raw_csi_signals(
                    n_samples=3,
                    save_path=figures_path / "raw_signals.png"
                )
                self.data_loader.analyze_signal_spectrum(
                    save_path=figures_path / "signal_spectrum.png"
                )
                self.data_loader.create_spectrograms(
                    save_path=figures_path / "spectrograms.png"
                )

                if self.labels is not None:
                    self.data_loader.analyze_activity_distribution(
                        save_path=figures_path / "activity_distribution.png"
                    )
            except Exception as viz_error:
                print(f"⚠️ Error creando visualizaciones: {viz_error}")
                print("Continuando sin guardar figuras...")

            self.logger.info(f"Datos cargados: {self.raw_data.shape}")
            print(f"✓ Datos cargados: {self.raw_data.shape}")

        except Exception as e:
            self.logger.error(f"Error cargando datos reales: {e}")
            print(f"❌ Error cargando datos: {e}")
            print("🔄 Generando datos sintéticos...")
            self._create_synthetic_data()

    def _create_synthetic_data(self):
        """
        Crea datos sintéticos para demostración
        """
        import numpy as np

        print("🔧 Generando datos sintéticos...")
        np.random.seed(self.config['modeling']['random_state'])

        # Parámetros
        n_samples = 10000
        n_antennas = 4
        n_subcarriers = 56
        activities = ['caminar', 'correr', 'sentado', 'parado', 'acostado']

        # Generar datos CSI sintéticos
        self.raw_data = np.zeros((n_samples, n_antennas, n_subcarriers))
        self.labels = np.zeros(n_samples // 100)

        print("📈 Generando patrones por actividad...")
        for i in range(n_samples):
            activity_idx = (i // 2000) % len(activities)

            # Patrones específicos por actividad
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
                    self.raw_data[i, ant, sub] = pattern + variation

            # Etiquetas
            if i % 100 == 0:
                self.labels[i // 100] = activity_idx

        # Crear data_loader sintético
        self.data_loader = WiFiDataLoader()
        self.data_loader.csi_data = self.raw_data
        self.data_loader.labels = self.labels
        self.data_loader.activities = activities

        print(f"✓ Datos sintéticos creados: {self.raw_data.shape}")
        print(f"✓ Actividades: {activities}")
        self.logger.info(f"Datos sintéticos generados: {self.raw_data.shape}")

    @timer_decorator
    def preprocess_data(self):
        """
        Paso 2: Preprocesamiento de datos
        """
        print("\n" + "=" * 60)
        print("PASO 2: PREPROCESAMIENTO DE DATOS")
        print("=" * 60)

        # Crear pipeline de preprocesamiento
        pipeline = CSIPreprocessingPipeline(
            sampling_rate=self.config['data']['sampling_rate']
        )

        print("🔧 Configurando pipeline de preprocesamiento...")

        # Configurar pasos según configuración
        pipeline.add_step('remove_dc', pipeline.preprocessor.remove_dc_component)

        filter_config = self.config['preprocessing']['bandpass_filter']
        pipeline.add_step('bandpass_filter', pipeline.preprocessor.apply_bandpass_filter, **filter_config)

        outlier_config = self.config['preprocessing']['outlier_removal']
        pipeline.add_step('remove_outliers', pipeline.preprocessor.remove_outliers, **outlier_config)

        pipeline.add_step('median_filter', pipeline.preprocessor.apply_median_filter, kernel_size=3)

        pipeline.add_step('normalize', pipeline.preprocessor.normalize_data,
                          method=self.config['preprocessing']['normalization'])

        pipeline.add_step('create_windows', pipeline.preprocessor.create_sliding_windows,
                          window_size=self.config['data']['window_size'],
                          overlap=self.config['data']['overlap'])

        print("⚙️ Ejecutando preprocesamiento...")
        # Ejecutar preprocesamiento
        self.processed_data, self.labels = pipeline.fit_transform(
            self.raw_data,
            self.labels,
            visualize=False  # Cambiar a True si quieres ver gráficos
        )

        # Guardar pipeline y datos procesados
        models_path = self.output_path / "models"
        data_path = self.output_path / "data"

        save_pickle(pipeline, models_path / "preprocessing_pipeline.pkl")
        save_pickle(self.processed_data, data_path / "processed_data.pkl")
        save_pickle(self.labels, data_path / "labels.pkl")

        # Log parámetros
        self.tracker.log_params({
            'preprocessing': self.config['preprocessing'],
            'data_config': self.config['data']
        })

        self.logger.info(f"Preprocesamiento completado: {self.processed_data.shape}")
        print(f"✓ Datos preprocesados: {self.processed_data.shape}")
        print(f"✓ Etiquetas procesadas: {self.labels.shape}")

    @timer_decorator
    def extract_features(self):
        """
        Paso 3: Extracción de características
        """
        print("\n" + "=" * 60)
        print("PASO 3: EXTRACCIÓN DE CARACTERÍSTICAS")
        print("=" * 60)

        # Crear extractor
        self.feature_extractor = CSIFeatureExtractor(
            sampling_rate=self.config['data']['sampling_rate']
        )

        print("🔍 Extrayendo características...")
        self.features = self.feature_extractor.extract_all_features(self.processed_data)

        # Análisis de características
        print("📊 Analizando importancia de características...")
        from src.feature_extraction import analyze_feature_importance, apply_pca_analysis

        features_selected, selected_names, importance_df = analyze_feature_importance(
            self.features, self.labels, self.feature_extractor.feature_names, k=50
        )

        print("📈 Aplicando análisis PCA...")
        features_pca, pca = apply_pca_analysis(self.features, n_components=0.95)

        # Guardar todo
        models_path = self.output_path / "models"
        data_path = self.output_path / "data"

        save_pickle(self.feature_extractor, models_path / "feature_extractor.pkl")
        save_pickle(self.features, data_path / "features.pkl")
        save_pickle(features_selected, data_path / "features_selected.pkl")
        save_pickle(importance_df, data_path / "feature_importance.pkl")
        save_pickle(pca, models_path / "pca_model.pkl")

        # Usar características seleccionadas para modelado
        self.features = features_selected

        # Log métricas
        self.tracker.log_metric('n_features_original', len(self.feature_extractor.feature_names))
        self.tracker.log_metric('n_features_selected', features_selected.shape[1])
        self.tracker.log_metric('n_features_pca', features_pca.shape[1])

        self.logger.info(f"Características extraídas: {self.features.shape}")
        print(f"✓ Características extraídas: {self.features.shape}")
        print(
            f"✓ Características seleccionadas: {features_selected.shape[1]} de {len(self.feature_extractor.feature_names)}")

    @timer_decorator
    def train_models(self):
        """
        Paso 4: Entrenamiento de modelos
        """
        print("\n" + "=" * 60)
        print("PASO 4: ENTRENAMIENTO DE MODELOS")
        print("=" * 60)

        # Crear sistema de modelos
        self.models = ActivityRecognitionModels(
            random_state=self.config['modeling']['random_state']
        )

        print("📊 Preparando datos para entrenamiento...")
        # Preparar datos
        X_train_scaled, X_val_scaled, X_test_scaled = self.models.prepare_data(
            self.features,
            self.labels,
            test_size=self.config['modeling']['test_size'],
            validation_size=self.config['modeling']['validation_size']
        )

        # Entrenar modelos tradicionales
        print("\n🌳 Entrenando Random Forest...")
        self.models.train_random_forest(n_estimators=100, max_depth=20)

        print("🔍 Entrenando SVM...")
        self.models.train_svm(C=1.0, kernel='rbf')

        print("🚀 Entrenando XGBoost...")
        try:
            self.models.train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1)
        except Exception as e:
            print(f"⚠️ XGBoost falló: {e}")
            print("🔄 Usando Random Forest adicional en lugar de XGBoost...")

            # Entrenar un Random Forest adicional como reemplazo
            from sklearn.ensemble import RandomForestClassifier
            rf_replacement = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['modeling']['random_state'],
                n_jobs=-1
            )

            rf_replacement.fit(self.models.X_train_scaled, self.models.y_train)

            train_acc = rf_replacement.score(self.models.X_train_scaled, self.models.y_train)
            val_acc = rf_replacement.score(self.models.X_val_scaled, self.models.y_val)
            test_acc = rf_replacement.score(self.models.X_test_scaled, self.models.y_test)

            # Guardar como "xgboost" para mantener compatibilidad
            self.models.models['xgboost'] = rf_replacement
            self.models.results['xgboost'] = {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'predictions_test': rf_replacement.predict(self.models.X_test_scaled),
                'predictions_proba': rf_replacement.predict_proba(self.models.X_test_scaled),
                'feature_importance': rf_replacement.feature_importances_,
                'model_type': 'RandomForest (XGBoost replacement)',
                'note': 'XGBoost falló, usando Random Forest como reemplazo'
            }

            print(f"✅ RF reemplazo - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    @timer_decorator
    def evaluate_models(self):
        """
        Paso 5: Evaluación y análisis
        """
        print("\n" + "=" * 60)
        print("PASO 5: EVALUACIÓN Y ANÁLISIS")
        print("=" * 60)

        print("📊 Evaluando todos los modelos...")
        # Evaluación completa
        results_df = self.models.evaluate_all_models()

        print("📈 Generando visualizaciones...")
        # Visualizaciones (sin mostrar en pantalla)
        import matplotlib
        matplotlib.use('Agg')  # Para guardar sin mostrar

        self.models.plot_training_history()
        self.models.plot_confusion_matrices()
        self.models.generate_classification_reports()

        # Análisis comparativo
        from src.models import compare_model_performance, analyze_feature_importance
        performance_df = compare_model_performance(self.models.results)

        if hasattr(self.feature_extractor, 'feature_names'):
            analyze_feature_importance(self.models.results, self.feature_extractor.feature_names)

        # Guardar resultados
        results_path = self.output_path / "results"
        results_df.to_csv(results_path / "model_comparison.csv", index=False)
        save_pickle(performance_df, results_path / "performance_analysis.pkl")

        # Encontrar mejor modelo
        if len(results_df) > 0:
            best_model_name = results_df.iloc[0]['Model']
            best_accuracy = results_df.iloc[0]['Test_Accuracy']

            self.tracker.log_metric('best_model_accuracy', best_accuracy)
            self.tracker.log_params({'best_model': best_model_name})

            print(f"\n🏆 Mejor modelo: {best_model_name}")
            print(f"📈 Accuracy: {best_accuracy:.4f}")
            self.logger.info(f"Mejor modelo: {best_model_name} - Accuracy: {best_accuracy:.4f}")

        print("✓ Evaluación completada")

    def generate_final_report(self):
        """
        Genera reporte final del proyecto
        """
        print("\n" + "=" * 60)
        print("GENERANDO REPORTE FINAL")
        print("=" * 60)

        # Información del proyecto
        project_info = {
            'Proyecto': 'Detección de Actividad Humana con WiFi Sensing',
            'Dataset': 'Sintético' if not self.data_path else f'Real: {self.data_path}',
            'Fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Datos_forma': str(self.raw_data.shape) if self.raw_data is not None else 'N/A',
            'Datos_procesados': str(self.processed_data.shape) if self.processed_data is not None else 'N/A',
            'Características': str(self.features.shape) if self.features is not None else 'N/A',
            'Modelos_entrenados': len(self.models.models) if self.models else 0,
            'Actividades': ', '.join(self.data_loader.activities) if self.data_loader and hasattr(self.data_loader,
                                                                                                  'activities') else 'N/A'
        }

        # Resultados
        results_summary = {}
        if self.models and self.models.results:
            best_accuracy = 0
            best_model = 'N/A'

            for model_name, results in self.models.results.items():
                if 'test_accuracy' in results:
                    accuracy = results['test_accuracy']
                    results_summary[f'{model_name}_accuracy'] = f"{accuracy:.4f}"

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name

            results_summary['mejor_modelo'] = best_model
            results_summary['mejor_accuracy'] = f"{best_accuracy:.4f}"

        # Crear reporte
        from src.utils import create_summary_report
        create_summary_report(
            project_info,
            results_summary,
            self.output_path / "project_summary.txt"
        )

        print(f"📄 Reporte final guardado en: {self.output_path}/project_summary.txt")

    def run(self):
        """
        Ejecuta todo el pipeline
        """
        start_time = time.time()

        print("🚀 INICIANDO PIPELINE DE DETECCIÓN DE ACTIVIDAD HUMANA")
        print("=" * 80)
        print(f"📁 Directorio de salida: {self.output_path}")
        print(f"🔧 Configuración: {self.config['modeling']['random_state']} (seed)")

        try:
            # Ejecutar pasos del pipeline
            self.load_data()
            self.preprocess_data()
            self.extract_features()
            self.train_models()
            self.evaluate_models()
            self.generate_final_report()

            # Finalizar experimento
            self.tracker.finalize()

            # Tiempo total
            total_time = time.time() - start_time

            print("\n" + "=" * 80)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print("=" * 80)
            print(f"⏱️ Tiempo total: {format_duration(total_time)}")
            print(f"📁 Resultados en: {self.output_path}")
            print(f"📊 Experimento: {self.tracker.experiment_path}")

            if self.models and self.models.results:
                best_result = max(
                    [(name, res['test_accuracy']) for name, res in self.models.results.items()
                     if 'test_accuracy' in res],
                    key=lambda x: x[1],
                    default=('N/A', 0)
                )
                print(f"🏆 Mejor modelo: {best_result[0]} ({best_result[1]:.4f})")

            self.logger.info(f"Pipeline completado en {format_duration(total_time)}")

        except Exception as e:
            print(f"\n❌ ERROR EN EL PIPELINE: {e}")
            print(f"📋 Traceback:\n{traceback.format_exc()}")
            self.logger.error(f"Error en pipeline: {e}", exc_info=True)
            raise


def main():
    """
    Función principal
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de Detección de Actividad Humana con WiFi Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                                    # Datos sintéticos
  python main.py --data data/raw/dataset.mat       # Con dataset real
  python main.py --config configs/config.json      # Con configuración
  python main.py --output results_custom/          # Directorio personalizado
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Ruta al archivo de configuración JSON/YAML'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Ruta al archivo de datos WI-MIR (.mat)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Directorio de salida (default: results)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Modo verboso'
    )

    args = parser.parse_args()

    # Mostrar información inicial
    print("🎯 DETECCIÓN DE ACTIVIDAD HUMANA - WiFi SENSING")
    print("=" * 50)

    if args.data:
        print(f"📁 Dataset: {args.data}")
    else:
        print("📊 Dataset: Sintético (para demostración)")

    if args.config:
        print(f"⚙️ Configuración: {args.config}")
    else:
        print("⚙️ Configuración: Por defecto")

    print(f"📂 Salida: {args.output}")
    print()

    try:
        # Crear y ejecutar pipeline
        pipeline = WiFiActivityDetectionPipeline(
            config_path=args.config,
            data_path=args.data,
            output_path=args.output
        )

        pipeline.run()

    except KeyboardInterrupt:
        print("\n⏹️ Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error fatal: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()