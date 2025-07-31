#!/usr/bin/env python3
"""
Script principal para ejecutar el pipeline completo con archivos .mat
====================================================================

Este script automatiza la ejecución del pipeline de detección de actividades
con todos los archivos .mat que tengas en la carpeta data/raw/
"""

import os
import sys
from pathlib import Path
import glob
import argparse

# Agregar src al path para importaciones
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))


def find_mat_files(data_raw_path="data/raw"):
    """
    Encuentra todos los archivos .mat en la carpeta

    Args:
        data_raw_path (str): Ruta a la carpeta con archivos raw

    Returns:
        list: Lista de archivos .mat encontrados
    """
    mat_files = glob.glob(os.path.join(data_raw_path, "*.mat"))
    print(f"📁 Archivos .mat encontrados: {len(mat_files)}")

    for i, file in enumerate(mat_files, 1):
        file_size = os.path.getsize(file) / (1024 ** 2)  # MB
        print(f"   {i}. {os.path.basename(file)} ({file_size:.1f} MB)")

    return mat_files


def select_file_interactive(mat_files):
    """
    Permite seleccionar un archivo interactivamente

    Args:
        mat_files (list): Lista de archivos disponibles

    Returns:
        str: Archivo seleccionado
    """
    if not mat_files:
        print("❌ No se encontraron archivos .mat en data/raw/")
        return None

    if len(mat_files) == 1:
        print(f"✓ Usando único archivo encontrado: {mat_files[0]}")
        return mat_files[0]

    print("\n🔍 Selecciona un archivo para procesar:")
    print("0. Procesar todos los archivos (uno por uno)")

    for i, file in enumerate(mat_files, 1):
        print(f"{i}. {os.path.basename(file)}")

    try:
        choice = int(input("\nIngresa tu elección (0-{}): ".format(len(mat_files))))

        if choice == 0:
            return "ALL"
        elif 1 <= choice <= len(mat_files):
            return mat_files[choice - 1]
        else:
            print("❌ Selección inválida")
            return None

    except ValueError:
        print("❌ Entrada inválida")
        return None


def run_pipeline_for_file(mat_file, output_base="results"):
    """
    Ejecuta el pipeline completo para un archivo específico

    Args:
        mat_file (str): Ruta al archivo .mat
        output_base (str): Directorio base para resultados

    Returns:
        bool: True si fue exitoso
    """
    from main import WiFiActivityDetectionPipeline

    # Crear nombre único para resultados basado en el archivo
    file_name = Path(mat_file).stem
    output_path = f"{output_base}_{file_name}"

    print(f"\n🚀 PROCESANDO: {os.path.basename(mat_file)}")
    print(f"📂 Resultados en: {output_path}")
    print("=" * 80)

    try:
        # Crear y ejecutar pipeline
        pipeline = WiFiActivityDetectionPipeline(
            data_path=mat_file,
            output_path=output_path
        )

        pipeline.run()

        print(f"\n✅ PROCESAMIENTO EXITOSO: {os.path.basename(mat_file)}")
        print(f"📊 Resultados guardados en: {output_path}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR procesando {os.path.basename(mat_file)}: {e}")
        print(f"📋 Detalles del error guardados en logs")
        return False


def run_batch_processing(mat_files, output_base="results"):
    """
    Procesa múltiples archivos en lote

    Args:
        mat_files (list): Lista de archivos a procesar
        output_base (str): Directorio base para resultados

    Returns:
        dict: Reporte de procesamiento
    """
    print(f"\n🔄 PROCESAMIENTO EN LOTE: {len(mat_files)} archivos")
    print("=" * 80)

    results = {
        'successful': [],
        'failed': [],
        'total': len(mat_files)
    }

    for i, mat_file in enumerate(mat_files, 1):
        print(f"\n[{i}/{len(mat_files)}] Procesando: {os.path.basename(mat_file)}")

        success = run_pipeline_for_file(mat_file, output_base)

        if success:
            results['successful'].append(mat_file)
        else:
            results['failed'].append(mat_file)

    # Reporte final
    print("\n" + "=" * 80)
    print("📊 REPORTE FINAL DE PROCESAMIENTO EN LOTE")
    print("=" * 80)
    print(f"✅ Exitosos: {len(results['successful'])}/{results['total']}")
    print(f"❌ Fallidos: {len(results['failed'])}/{results['total']}")

    if results['failed']:
        print(f"\n⚠️ Archivos con errores:")
        for failed_file in results['failed']:
            print(f"   - {os.path.basename(failed_file)}")

    return results


def create_batch_config(output_path="configs/batch_config.json"):
    """
    Crea configuración optimizada para procesamiento en lote
    """
    from utils import save_config, get_default_config

    # Configuración optimizada para lote
    config = get_default_config()

    # Ajustes para procesamiento rápido
    config['preprocessing']['bandpass_filter']['order'] = 3  # Filtro más rápido
    config['modeling']['test_size'] = 0.2  # Menos datos para test
    config['modeling']['validation_size'] = 0.15  # Menos datos para validación

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_config(config, output_path)
    print(f"📝 Configuración de lote guardada en: {output_path}")

    return config


def main():
    """
    Función principal
    """
    parser = argparse.ArgumentParser(
        description="Ejecutor del pipeline WiFi Activity Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python execute_pipeline.py                          # Selección interactiva
  python execute_pipeline.py --file data/raw/dataset1.mat  # Archivo específico
  python execute_pipeline.py --all                    # Todos los archivos
  python execute_pipeline.py --batch --output results_batch/  # Lote con directorio custom
        """
    )

    parser.add_argument(
        '--file',
        type=str,
        help='Archivo .mat específico a procesar'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Procesar todos los archivos .mat encontrados'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Modo de procesamiento en lote (mismo que --all)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Directorio base para resultados (default: results)'
    )

    parser.add_argument(
        '--raw-path',
        type=str,
        default='data/raw',
        help='Ruta a la carpeta con archivos .mat (default: data/raw)'
    )

    args = parser.parse_args()

    print("🎯 EJECUTOR DEL PIPELINE WIFI ACTIVITY DETECTION")
    print("=" * 60)

    # Verificar estructura del proyecto
    if not os.path.exists('main.py'):
        print("❌ Error: No se encuentra src/main.py")
        print("   Asegúrate de ejecutar desde el directorio raíz del proyecto")
        sys.exit(1)

    if not os.path.exists(args.raw_path):
        print(f"❌ Error: No se encuentra la carpeta {args.raw_path}")
        print("   Crea la carpeta y coloca tus archivos .mat allí")
        sys.exit(1)

    # Buscar archivos .mat
    mat_files = find_mat_files(args.raw_path)

    if not mat_files:
        print(f"\n❌ No se encontraron archivos .mat en {args.raw_path}")
        print("   Coloca tus archivos .mat en esa carpeta y vuelve a intentar")
        sys.exit(1)

    # Determinar qué procesar
    if args.file:
        # Archivo específico
        if not os.path.exists(args.file):
            print(f"❌ Error: Archivo no encontrado: {args.file}")
            sys.exit(1)

        print(f"📄 Procesando archivo específico: {args.file}")
        run_pipeline_for_file(args.file, args.output)

    elif args.all or args.batch:
        # Todos los archivos
        print(f"📦 Procesando todos los archivos en modo lote")
        create_batch_config()  # Crear configuración optimizada
        run_batch_processing(mat_files, args.output)

    else:
        # Selección interactiva
        selected = select_file_interactive(mat_files)

        if selected is None:
            print("❌ No se seleccionó ningún archivo")
            sys.exit(1)
        elif selected == "ALL":
            print(f"📦 Procesando todos los archivos")
            create_batch_config()
            run_batch_processing(mat_files, args.output)
        else:
            print(f"📄 Procesando archivo seleccionado: {selected}")
            run_pipeline_for_file(selected, args.output)


if __name__ == "__main__":
    main()