# WiFi Activity Detection

Proyecto para detección de actividad humana usando señales WiFi CSI del dataset WI-MIR.

## Estructura del Proyecto

```
wifi-activity-detection/
│
├── data/
│   ├── raw/                    # Dataset WI-MIR original
│   └── processed/              # Datos preprocesados
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── utils.py
│
├── results/
│   ├── figures/
│   ├── models/
│   └── metrics/
│
└── requirements.txt
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

1. Coloca el dataset WI-MIR en la carpeta `data/raw/`
2. Ejecuta los notebooks en orden numérico
3. Los resultados se guardarán en la carpeta `results/`
