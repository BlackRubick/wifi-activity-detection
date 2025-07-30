"""
Script de debug para analizar exactamente qué está pasando con tus datos
======================================================================

Ejecuta esto para ver exactamente qué contiene tu archivo
"""

from scipy.io import loadmat
import numpy as np


def debug_mat_file(filepath):
    """
    Debug completo de un archivo .mat
    """
    print(f"🔍 DEBUGGING: {filepath}")
    print("=" * 60)

    # Cargar archivo
    data = loadmat(filepath)
    clean_data = {k: v for k, v in data.items() if not k.startswith('__')}

    print(f"📊 Variables: {list(clean_data.keys())}")

    # Analizar Raw_Cell_Matrix
    if 'Raw_Cell_Matrix' in clean_data:
        cell_matrix = clean_data['Raw_Cell_Matrix']
        print(f"\n📦 Raw_Cell_Matrix:")
        print(f"   Shape: {cell_matrix.shape}")
        print(f"   Dtype: {cell_matrix.dtype}")
        print(f"   Size: {cell_matrix.size}")

        print(f"\n🔍 Explorando elementos:")

        # Explorar elementos
        for i in range(min(20, cell_matrix.size)):
            try:
                if cell_matrix.ndim == 2:
                    element = cell_matrix[i, 0]
                else:
                    element = cell_matrix.flat[i]

                print(f"   [{i:2d}] Tipo: {type(element)}")

                if isinstance(element, np.ndarray):
                    print(f"        Shape: {element.shape}")
                    print(f"        Dtype: {element.dtype}")
                    print(f"        Size: {element.size}")

                    if element.size > 0:
                        try:
                            print(f"        Std: {np.std(element):.6f}")
                            print(f"        Min: {np.min(element):.6f}")
                            print(f"        Max: {np.max(element):.6f}")

                            # ¿Es este el CSI?
                            if element.size > 1000 and np.std(element) > 0:
                                print(f"        🎯 ¡POSIBLE CSI DETECTADO!")

                                # Mostrar más detalles
                                if len(element.shape) == 3:
                                    print(
                                        f"        📊 3D: [tiempo={element.shape[0]}, ant={element.shape[1]}, sub={element.shape[2]}]")
                                elif len(element.shape) == 2:
                                    print(f"        📊 2D: [dim1={element.shape[0]}, dim2={element.shape[1]}]")

                                # Esta es la línea de código que necesitas
                                print(f"        💡 CÓDIGO: csi_data = raw_cell[{i}, 0]")
                                return i  # Retornar índice del CSI

                        except Exception as e:
                            print(f"        ❌ Error calculando stats: {e}")

                elif isinstance(element, (str, bytes)):
                    print(f"        Contenido: {str(element)[:100]}...")

                else:
                    print(f"        Valor: {str(element)[:100]}")

            except Exception as e:
                print(f"   [{i:2d}] ❌ Error: {e}")

    return None


# Test rápido
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Buscar archivo automáticamente
        import glob

        files = glob.glob("data/raw/*.mat")
        if files:
            filepath = files[0]
            print(f"📁 Usando: {filepath}")
        else:
            print("❌ No se encontró archivo .mat")
            print("Uso: python debug_script.py archivo.mat")
            exit(1)

    csi_index = debug_mat_file(filepath)

    if csi_index is not None:
        print(f"\n🎯 CSI ENCONTRADO EN ÍNDICE: {csi_index}")
        print(f"\n🔧 SOLUCIÓN:")
        print(f"Modifica src/data_loader.py, método _extract_csi_and_labels")
        print(f"Cambia la línea donde dice:")
        print(f"   element = raw_cell[i, 0]")
        print(f"Por:")
        print(f"   element = raw_cell[{csi_index}, 0]")
        print(f"")
        print(f"O mejor aún, usa este código completo:")

        code = f'''
def _extract_csi_and_labels(self):
    """
    Extrae automáticamente datos CSI y etiquetas del dataset cargado
    """
    if self.raw_data is None:
        return

    print("🔍 Extrayendo datos de cell array...")

    # Buscar Raw_Cell_Matrix
    if 'Raw_Cell_Matrix' in self.raw_data:
        raw_cell = self.raw_data['Raw_Cell_Matrix']
        print(f"📦 Cell array encontrado: {{raw_cell.shape}}")

        # USAR EL ÍNDICE DETECTADO DIRECTAMENTE
        try:
            self.csi_data = raw_cell[{csi_index}, 0]
            print(f"✅ Datos CSI extraídos del índice {csi_index}")
            print(f"   Shape: {{self.csi_data.shape}}")
            print(f"   Dtype: {{self.csi_data.dtype}}")
            print(f"   Std: {{np.std(self.csi_data):.6f}}")

        except Exception as e:
            print(f"❌ Error extrayendo CSI: {{e}}")
            return

    # Crear etiquetas sintéticas
    if self.csi_data is not None:
        if len(self.csi_data.shape) == 3:
            n_time_samples = self.csi_data.shape[0]
            n_labels = max(1, n_time_samples // 100)
        elif len(self.csi_data.shape) == 2:
            n_labels = max(1, self.csi_data.shape[0] // 100)
        else:
            n_labels = 50

        self.labels = np.array([i % 3 for i in range(n_labels)])
        self.activities = ['Caminar', 'Estar_Quieto', 'Movimiento_Manos']

        print(f"✅ Etiquetas sintéticas: {{n_labels}} etiquetas, 3 actividades")
'''

        print(code)

    else:
        print(f"\n❌ No se detectó CSI automáticamente")
        print(f"💡 Revisa manualmente los elementos mostrados arriba")