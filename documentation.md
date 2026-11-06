# Guía técnica de la entrega

## Alcance y flujo de datos

Consultar el [README](README.md) para instalar, preparar los datos y ejecutar
el flujo principal. El algoritmo utiliza TC y anotaciones alineadas: hígado
(1), tumor (2) y fondo (0). La ROI hepática incluye las etiquetas 1 y 2.
Las anotaciones tumorales permiten elegir cortes y evaluar predicciones; Otsu
segmenta intensidades dentro de la ROI hepática conocida.

El orden principal es: carga de configuración → selección de pacientes y
cortes → recorte hepático → ventana HU → suavizado → realce opcional →
segmentación → morfología → máscaras, figuras y evaluación.

La configuración final es Otsu sin CLAHE, sobre 131 pacientes y hasta cinco
cortes por paciente. El muestreo depende de las referencias y no constituye
un muestreo ciego. La prueba en los 131 pacientes fue realizada en otra
computadora, según lo informado por el equipo; el repositorio no contiene sus
resultados. El ajuste previo utilizó diez pacientes del mismo conjunto.

## Entradas, configuración y presets

`config.yaml` contiene parámetros portables; `config.local.yaml`, si existe
junto al YAML solicitado, lo reemplaza completamente. Se conservan claves
`data_dir`, `output_dir`, `selection`, `preprocessing`, `segmentation`,
`detection`, `features`, `compression`, `reconstruction` y `edge_analysis`.
Las rutas se resuelven respecto de la ubicación del YAML. La carga crea la
carpeta de salida cuando hace falta.

`selection.case_ids` permite fijar IDs y prevalece sobre `n_cases`. El archivo
final no contiene esa lista. `slices_per_case` admite 3–5. La estrategia
`stratified` busca distintos tamaños tumorales y un corte negativo, con
separación física mínima; los roles que no puedan completarse se informan.

`run_tpi.py` genera nuevamente la selección antes de segmentar. Por compatibilidad,
si `data_dir` no tiene pares, busca en `data/Selected_database`; esa carpeta no
se entrega y no altera el requisito de cantidad de pacientes. Los demás flujos
utilizan la ruta configurada. No se proporciona ni se copia ningún dataset.

Los recursos de `config/presets/` se distribuyen con el código:

- `controls.json`: alternativas históricas HU, Otsu con CLAHE y Otsu sin CLAHE,
  utilizadas por el selector de presets de la GUI. El nombre histórico `default`
  no corresponde a la configuración final.
- `preset_growth.yaml`, `preset_contour.yaml`, `preset_graphcut.yaml`: ajustes
  parciales para **Cargar preset local YAML**. No son configuraciones completas
  de lote. Para un lote, copiar el config principal y sustituir las secciones
  del preset; mantener explícitamente rutas, selección y demás parámetros.

Los detectores locales admiten `local_pipeline: configured`, que aplica el
preprocesamiento configurado y la morfología final, o `raw_hu` para operar sobre
HU originales. Una transformación no lineal cambia la interpretación de las
intensidades; no atribuir a ambas variantes los mismos resultados.

## Scripts entregados

Todos los comandos se ejecutan desde la raíz con el entorno activado. Los que
admiten argumentos muestran sus opciones con `--help`. `explore_dataset.py`,
`select_cases.py` y `preprocess_selected.py` leen directamente el config de la
raíz y su posible reemplazo local; no tienen parser de argumentos.

| Script | Función y requisitos |
| --- | --- |
| `gui.py` | Interfaz Tk: carga, navegación, ajustes, segmentación, cuantificación y resultados. |
| `run_tpi.py` | Selecciona cortes y ejecuta el flujo principal con un ID temporal nuevo. |
| `select_slices_2d.py` | Genera la selección estratificada con lecturas 2D de máscaras. |
| `explore_dataset.py` | Inventario volumétrico de una muestra de hasta `candidate_pool_size` casos, contexto y figuras. Por defecto 15, no 131. |
| `select_cases.py` | Selección alternativa balanceada desde `dataset_inventory.csv`; reemplaza el CSV de selección. No usar para reproducir el lote final de 131 casos. |
| `preprocess_selected.py` | Figuras de los cortes seleccionados antes y después del procesamiento. |
| `compare_filters.py` | Compara filtros bilateral, gaussiano y media sobre el primer corte seleccionado. |
| `run_segmentation.py` | Segmentación de una selección existente; `--run-id` obligatorio. `--sweep` habilita el barrido básico de umbrales y morfología. |
| `run_detection.py` | Ejecutor HU usado por el despachador; admite corrida individual y opciones de comparación interna. El benchmark exige insumos adicionales y no corresponde a la prueba final de 131 casos. |
| `run_local_segmentation.py` | Ejecuta un preset local o híbrido completo con `--config` y `--run-id`; también lo utiliza el despachador. |
| `extract_features.py` | Geometría y GLCM en regiones de referencia; `--levels 8 16 32` evalúa sensibilidad a cuantización. |
| `plot_features.py` | Figuras de textura por corte y paciente a partir de tablas extraídas. |
| `evaluate_lesions.py` | Correspondencias y métricas por lesión para una corrida y configuración. |
| `analyze_agreement.py` | Bland–Altman y correlaciones a partir de `lesion_matches.csv`. |
| `compare_methods.py` | Comparación pareada por paciente de configuraciones de una corrida, Wilcoxon y Holm. |
| `analyze_morphology_3d.py` | Volumen, componentes y diámetros de las máscaras de referencia de los casos seleccionados. |
| `evaluate_morphology_3d.py` | Apila predicciones 2D a lo largo de volúmenes seleccionados y las compara con referencia. |
| `run_compression.py` | Control de 8 bits, PNG, Huffman, RLE y compresión con pérdida DCT; requiere selección previa. |
| `run_reconstruction.py` | Proyecciones simuladas, reconstrucciones y comparación; requiere selección previa. |
| `check_requirements.py` | Comprueba versiones de dependencias directas. |
| `build_icons.py` | Regenera recursos PNG e ICO; los recursos ya están incluidos. |

`PSIB.bat` prepara el entorno y abre la GUI en Windows; `liver_tumor.sh` activa
un entorno existente en Linux. `create_shortcut.ps1` crea accesos directos de
Windows; `psib.desktop.template` sirve como plantilla para un acceso Linux y
requiere reemplazar `<project-root>`. Los accesos generados son locales.

## Salidas y evaluación

Las salidas se generan en `results/` y no se versionan:

- `selected_cases_slices.csv`: IDs de paciente, índice axial, rol, semilla y
  criterios de selección. Se regenera al ejecutar el flujo principal.
- `dataset_inventory.csv`, `candidate_pool.csv`, `dataset_context_table.csv`
  y figuras de contextualización: corresponden al inventario muestreado.
- `segmentation/ID/`: configuración efectiva, identificación de variantes,
  máscaras, figuras y métricas por corte, por paciente y resumen.
- `features/`: geometría, texturas por región/paciente, diferencias pareadas y
  sensibilidad a niveles GLCM. Se calculan sobre máscaras de referencia.
- `segmentation/ID/lesion_eval/`: correspondencias, sensibilidad y falsos
  positivos; `agreement/` contiene el análisis de acuerdo posterior.
- `compression/ID/` y `reconstruction/ID/`: métricas, resúmenes y figuras de
  cada experimento. No se reutilizan IDs existentes.

En Otsu individual, la configuración es `cfg_000`. Para otras rutas, consultar
`configurations.csv`: los IDs locales e híbridos son diferentes. El argumento
`best`, disponible en evaluadores, selecciona por Dice sobre los resultados de
la corrida y no constituye validación independiente.

Dice e IoU comparan píxeles; cobertura de referencia es intersección dividida
por área tumoral y no equivale a sensibilidad por lesión. La evaluación por
lesiones incluye correspondencias que consideran divisiones/fusiones y una
salida adicional de emparejamiento uno a uno. El umbral IoU debe informarse al
comparar resultados. El config final mantiene `lesion_eval.min_iou: 0.0`;
puede indicarse otro mediante `--min-iou`, sin atribuirle métricas de otro umbral.

Las observaciones de un paciente no son réplicas independientes. Las tablas por
paciente y comparaciones pareadas ayudan a evitar interpretar muchos cortes
como muchos pacientes. La textura descriptiva de regiones anotadas no demuestra
capacidad diagnóstica autónoma.

## Extensiones 3D y límites

Después de generar la selección:

```bash
python scripts/analyze_morphology_3d.py                 # opcional: --cases 3 10 --run-id NOMBRE
python scripts/evaluate_morphology_3d.py --run-id ID_CORRIDA --configuration-id cfg_000
# opcional: --cases 3 10 · --min-component-mm3 50 · --min-slices 2 · --save-masks
```

`analyze_morphology_3d.py` escribe en `results/morphology_3d/<fecha-hora>/`; las
columnas `largest_*` describen la lesión más grande y `max_axial_feret_all_mm`
el máximo sobre todas. Los casos sin tumor dejan los tamaños en NaN.

`evaluate_morphology_3d.py` escribe además `morphology_3d_prediction_lesions.csv`
(una fila por lesión de referencia: detectada, cobertura, Dice de lesión) y
`morphology_3d_prediction_by_size.csv` (detección y Dice por diámetro axial
<10, 10–20, ≥10 y ≥20 mm). Las lesiones se cuentan con conectividad 26, para
que fragmentos de la anotación unidos por una arista o vértice no cuenten como
lesiones aparte. Una lesión está *detectada* si la toca alguna componente
predicha y su IoU de lesión es ≥ `lesion_eval.min_iou`; está *cubierta* si al
menos el 50 % de sus vóxeles fue predicho. Usar la tasa de cubiertas: una región
falsa grande que solo roza una lesión pequeña la marca como detectada.

Límites a declarar: la ROI hepática es la de referencia (`liver_roi=reference`),
por lo que no hay falsos positivos fuera del hígado y el Dice es optimista. El
filtro 3D (`morphology3d_prediction`: `min_component_voxels`, y opcionalmente
`min_component_mm3` y `min_slices`) se aplica solo a la predicción; el umbral en
vóxeles cambia de tamaño físico con el espaciado, el de mm³ no.

El primer comando mide máscaras de referencia; el segundo aplica el segmentador
2D a los cortes de los volúmenes y los apila. Es una extensión exploratoria, no
un modelo 3D independiente. Puede requerir bastante más tiempo y memoria que
cinco cortes por paciente. El Feret axial máximo, el Feret 3D y la diagonal de
la caja envolvente son medidas distintas.

La reconstrucción simula una adquisición desde imágenes ya reconstruidas.
La compresión parte de intensidades mapeadas a 8 bits y compara con ese control;
no confundir sus pérdidas con las del HU original ni con una adquisición clínica.

## Organización interna y pruebas

`src/psib/` agrupa carga/configuración, dataset, preprocesamiento, segmentadores,
características, métricas, evaluación, compresión, reconstrucción, visualización
y estado de la GUI. `serialization.py` escribe diagnósticos JSON de manera
atómica y acepta valores NumPy. Los scripts coordinan estas funciones.

`tests/` conserva pruebas unitarias e integraciones sintéticas, incluidas
segmentación local/híbrida, compresión y reconstrucción. Los fixtures versionados
son entradas de prueba, no resultados clínicos. La prueba de selección de entrega
simula 131 pacientes para verificar que ninguno quede excluido por IDs fijos.

```bash
python -m unittest discover -s tests -v
```

Datos, resultados, perfiles de GUI, experimentos históricos y entornos virtuales
están ignorados. El árbol final no requiere la carpeta de desarrollo original.
Se conserva el historial Git sin reescribirlo; no se incluyen informes ni
presentación oral en esta preparación del repositorio.
