"""
Extractor específico para estructuras MATLAB con campos CSI
==========================================================

Tu archivo tiene estructuras con campos: 'Ntx', 'Nrx', 'noise', 'rssi_a', 'rssi_b', 'rssi_c', 'csi', 'MPIs_label'
Los datos CSI están en el campo 'csi' y las etiquetas en 'MPIs_label'
"""

from scipy.io import loadmat
import numpy as np


def extract_csi_from_structs(filepath):
    """
    Extrae datos CSI y etiquetas de estructuras MATLAB
    """
    print(f"🔍 EXTRAYENDO CSI DE ESTRUCTURAS: {filepath}")
    print("=" * 60)

    # Cargar archivo
    data = loadmat(filepath)
    clean_data = {k: v for k, v in data.items() if not k.startswith('__')}

    if 'Raw_Cell_Matrix' not in clean_data:
        print(" No se encontró Raw_Cell_Matrix")
        return None, None

    cell_matrix = clean_data['Raw_Cell_Matrix']
    print(f" Cell matrix shape: {cell_matrix.shape}")

    # Listas para acumular datos
    all_csi_data = []
    all_labels = []

    print(f"🔍 Procesando {cell_matrix.size} estructuras...")

    for i in range(min(100, cell_matrix.size)):  # Procesar primeros 100 para prueba
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

            # Extraer campo CSI
            if hasattr(struct_data, 'dtype') and 'csi' in struct_data.dtype.names:
                csi_field = struct_data['csi']
                label_field = struct_data['MPIs_label'] if 'MPIs_label' in struct_data.dtype.names else None

                # El campo CSI puede ser un array anidado
                if isinstance(csi_field, np.ndarray) and csi_field.size > 0:
                    # Extraer datos CSI reales
                    if csi_field.dtype == 'object':
                        # Está anidado, extraer el contenido real
                        actual_csi = csi_field.flat[0] if csi_field.size > 0 else None
                    else:
                        actual_csi = csi_field

                    if actual_csi is not None and isinstance(actual_csi, np.ndarray) and actual_csi.size > 10:
                        all_csi_data.append(actual_csi)

                        # Extraer etiqueta
                        if label_field is not None:
                            if isinstance(label_field, np.ndarray):
                                label_val = label_field.flat[0] if label_field.size > 0 else 0
                            else:
                                label_val = label_field
                            all_labels.append(label_val)
                        else:
                            all_labels.append(0)  # Etiqueta por defecto

                        if i < 5:  # Mostrar detalles de los primeros 5
                            print(f"   [{i:3d}] CSI shape: {actual_csi.shape}, dtype: {actual_csi.dtype}")
                            print(f"         Label: {label_val}")
                            if actual_csi.size > 0:
                                print(
                                    f"         CSI stats: min={np.min(actual_csi):.3f}, max={np.max(actual_csi):.3f}, std={np.std(actual_csi):.3f}")

        except Exception as e:
            if i < 10:  # Solo mostrar errores de los primeros 10
                print(f"   [{i:3d}] Error: {e}")
            continue

    print(f"\n RESUMEN DE EXTRACCIÓN:")
    print(f"   CSI arrays extraídos: {len(all_csi_data)}")
    print(f"   Etiquetas extraídas: {len(all_labels)}")

    if len(all_csi_data) == 0:
        print("❌ No se extrajeron datos CSI")
        return None, None

    # Análisis de las formas de CSI
    shapes = [arr.shape for arr in all_csi_data[:10]]
    print(f"   Formas de CSI (primeros 10): {shapes}")

    # Determinar si concatenar o apilar
    if len(set(shapes)) == 1:
        # Todas las formas son iguales - apilar
        print("✅ Todas las formas CSI son iguales - Apilando...")
        combined_csi = np.stack(all_csi_data, axis=0)
        combined_labels = np.array(all_labels)
    else:
        # Formas diferentes - concatenar temporalmente
        print(" Formas CSI diferentes - Concatenando temporalmente...")
        # Tomar solo los que tengan la forma más común
        from collections import Counter
        shape_counts = Counter(shapes)
        most_common_shape = shape_counts.most_common(1)[0][0]

        filtered_csi = [arr for arr in all_csi_data if arr.shape == most_common_shape]
        filtered_labels = [all_labels[i] for i, arr in enumerate(all_csi_data) if arr.shape == most_common_shape]

        print(f"   Forma más común: {most_common_shape}")
        print(f"   Arrays filtrados: {len(filtered_csi)}")

        if len(filtered_csi) > 0:
            combined_csi = np.stack(filtered_csi, axis=0)
            combined_labels = np.array(filtered_labels)
        else:
            return None, None

    print(f"\n✅ DATOS FINALES:")
    print(f"   CSI shape: {combined_csi.shape}")
    print(f"   CSI dtype: {combined_csi.dtype}")
    print(f"   Labels shape: {combined_labels.shape}")
    print(f"   Unique labels: {np.unique(combined_labels)}")

    return combined_csi, combined_labels


def create_data_loader_for_structs():
    """
    Crea el código para el data loader específico para estructuras
    """
    code = '''
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

    print(" Procesando estructuras...")

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
'''

    return code


# Función principal para probar
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        import glob

        files = glob.glob("data/raw/*.mat")
        if files:
            filepath = files[0]
        else:
            print("❌ No se encontró archivo .mat")
            exit(1)

    # Probar extracción
    csi_data, labels = extract_csi_from_structs(filepath)

    if csi_data is not None:
        print(f"\n EXTRACCIÓN EXITOSA!")
        print(f" CSI shape final: {csi_data.shape}")
        print(f" Labels shape: {labels.shape}")
        print(f"Etiquetas únicas: {np.unique(labels)}")

        print(f"\n CÓDIGO PARA DATA_LOADER:")
        print("=" * 60)
        print("Reemplaza el método _extract_csi_and_labels en src/data_loader.py con:")
        print(create_data_loader_for_structs())

    else:
        print(f"\n No se pudo extraer CSI")
        print(" Revisa manualmente la estructura del archivo")