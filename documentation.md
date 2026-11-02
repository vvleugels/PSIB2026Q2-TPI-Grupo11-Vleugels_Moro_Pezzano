# Documentación del programa

## 1. Propósito y estado actual

El proyecto permite explorar tomografías de hígado etiquetadas, seleccionar
casos y cortes axiales, preprocesarlos y aplicar métodos clásicos de
segmentación de posibles tumores. También incluye métricas Dice e IoU.

Los flujos se ejecutan mediante scripts Python. `src/psib/` contiene la
lógica reutilizable y `scripts/` los puntos de entrada ejecutables.

| Capacidad | Estado |
| --- | --- |
| Descubrimiento y carga de pares CT/máscara NIfTI | Implementado |
| Inventario y selección reproducible de casos y cortes | Implementado |
| Preprocesamiento y exportación de figuras | Implementado |
| Segmentación 2D y posprocesamiento | Implementado como API Python |
| Dice e IoU | Implementados como API Python |
| Características geométricas y textura GLCM 2D | Implementadas como API Python |
| Tablas de características para cortes seleccionados | Implementadas como script |
| Boxplots de textura entre regiones de referencia | Implementados como script |
| Resumen de textura por caso | Implementado como tabla y gráfico descriptivo |
| Diferencias de textura emparejadas por caso | Implementadas como tablas descriptivas |
| Volumen, burden, componentes y diámetros 3D desde máscaras de referencia | Implementados en la exploración volumétrica opcional |
| Análisis morfológico 3D de los casos seleccionados, exportación de medidas por lesión y superficies | Implementado como flujo opcional invocable desde el GUI |
| Evaluación por lesión, errores geométricos y detección por tamaño en cortes 2D | Implementada como flujo posterior a la segmentación |
| Estimación automática 3D por apilado de predicciones axiales | Implementada como análisis opcional de volúmenes completos |
| Concordancia Bland–Altman y comparación estadística por caso | Implementadas como análisis posteriores |
| Ejecutor de segmentación de todos los cortes y exportación de máscaras/métricas | Implementado |
| Barrido de 252 configuraciones y comparación exploratoria | Implementado |
| Comparación de filtros sobre un corte y exportación sin GUI | Implementado |
| Selección de roles tumorales con separación física en mm | Implementada; `area_top` mantiene el método previo |
| Tabla de contexto HU, burden y tamaños de lesión | Implementada en el inventario etiquetado |
| Experimento de compresión reversible e irreversible | Implementado sobre cortes seleccionados |
| Simulación de adquisición y reconstrucción tomográfica | Implementada; no es adquisición clínica real |
| Lanzador con fallback de datos e identificadores temporales | Implementado |
| Interfaz gráfica para ejecutar selección 2D, segmentación, análisis 3D, compresión y reconstrucción | Implementada como lanzador de escritorio |
| Superposición de contornos de predicción y referencia | Disponible como API Python |
| Evaluación independiente con otra base | Pendiente |

El flujo principal usa lectura por cortes 2D para limitar el consumo de memoria.
El inventario exploratorio calcula volumen tumoral y componentes desde las
máscaras de referencia. El flujo morfológico opcional descrito a continuación
agrega medidas por componente y figuras de superficie sobre la referencia. La
estimación automática 3D opcional se describe en la sección 6; apila
predicciones del segmentador 2D y no es una segmentación 3D aprendida.

### Alcance mínimo y máximo

El **objetivo mínimo** es segmentar y cuantificar en 2D los tumores de los
cortes axiales seleccionados en los casos etiquetados del subconjunto local.
El objetivo final comprende 10 casos; la configuración actual toma 1 caso para
probar el flujo rápidamente. Restaurar `selection.n_cases: 10` antes de
producir los resultados comparativos finales. Se solicitan 5 cortes por caso y
separa los índices por un mínimo de 2 cortes; si esa separación lo impide,
puede seleccionar menos. En esos cortes se calculan características
geométricas y textura GLCM 2D. La segmentación compara umbrales globales, Otsu
y estadísticos, además de variantes morfológicas, usando Dice e IoU frente a
las máscaras de referencia.

El **objetivo máximo** agrega morfología descriptiva 3D de las máscaras de
referencia: volumen tumoral y hepático, carga tumoral, componentes 6-conexos,
Feret axial y 3D, diámetro esférico equivalente y diagonal de caja por lesión.
El diámetro máximo informado es el Feret axial máximo (tipo RECIST); Feret 3D
queda como medida secundaria y la diagonal de caja se conserva con nombre
explícito porque sobreestima el diámetro. El GUI muestra Feret axial. La
Como extensión opcional, el segmentador 2D se aplica a todos los cortes de cada
volumen seleccionado y se apila para una estimación automática 3D. La evaluación
por lesión, la textura y la comparación estadística siguen siendo 2D sobre
cortes seleccionados.

## 2. Estructura y responsabilidades

| Ubicación | Responsabilidad |
| --- | --- |
| `src/psib/config.py` | Lectura del YAML, precedencia local y resolución de rutas |
| `src/psib/data_io.py` | Descubrimiento, orientación, carga por volumen y lector por cortes |
| `src/psib/dataset.py` | Inventario, terciles de volumen tumoral y selección |
| `src/psib/preprocessing.py` | Ventana HU, filtros, realce y recorte hepático |
| `src/psib/segmentation.py` | Umbrales 2D, morfología y componentes conectados |
| `src/psib/metrics.py` | Dice e IoU entre máscaras binarias |
| `src/psib/features.py` | Geometría y textura GLCM 2D, con entropía y ASM |
| `src/psib/lesion_eval.py` | Correspondencia split/merge y error geométrico por lesión |
| `src/psib/morphology3d.py` | Volumen físico, Feret 3D/axial y diámetros equivalentes |
| `src/psib/compression.py` | Entropía, Huffman, RLE y codec didáctico DCT/JPEG |
| `src/psib/reconstruction.py` | Atenuación, padding circular seguro y reconstrucción Radon |
| `scripts/extract_features.py` | Tablas de características para cortes seleccionados |
| `scripts/plot_features.py` | Boxplots descriptivos de propiedades GLCM por tejido |
| `src/psib/visualization.py` | Contornos de predicción y referencia sobre CT |
| `scripts/run_tpi.py` | Selección 2D y segmentación base con ID temporal |
| `scripts/select_slices_2d.py` | Selección con lectura de máscaras por planos |
| `scripts/run_segmentation.py` | Ejecuciones base y barrido de parámetros |
| `scripts/evaluate_lesions.py` | Evaluación de detección y geometría vs. referencia |
| `scripts/evaluate_morphology_3d.py` | Estimación 3D apilando predicciones axiales de la segmentación 2D |
| `scripts/analyze_agreement.py` | Bland–Altman y correlaciones de medidas emparejadas |
| `scripts/compare_methods.py` | Wilcoxon por caso y corrección Holm |
| `scripts/run_compression.py` | Comparación de compresión y segmentación por corte |
| `scripts/run_reconstruction.py` | Simulación tomográfica y comparación de métodos |
| `scripts/check_requirements.py` | Verificación de dependencias directas del lanzador |
| `scripts/compare_filters.py` | Comparación de mediana, gaussiano y bilateral |
| `scripts/explore_dataset.py` | Inventario volumétrico y figuras de contexto |
| `scripts/analyze_morphology_3d.py` | CSV y superficies 3D de los casos seleccionados |
| `src/psib/gui_model.py` | Estado de sidebar y persistencia del perfil, sin dependencias Tk |
| `src/psib/app_icon.py` | Rutas de iconos y carga defensiva con fallback |
| `scripts/gui.py` | GUI con sidebar seccionado, visor compartido y ejecución en segundo plano |
| `tests/test_gui_model.py` | Estado, expansión, persistencia y clamp de ancho del sidebar |
| `tests/test_app_icon.py` | Construcción determinista, tamaños ICO y fallback sin display |
| `scripts/build_icons.py` | Derivación determinista de PNG, ICO y logo desde el master |
| `scripts/create_shortcut.ps1` | Accesos directos locales Windows, sin elevación |
| `scripts/psib.desktop.template` | Plantilla opcional para escritorios Linux |
| `scripts/select_cases.py` | Selección por terciles de volumen tumoral |
| `scripts/preprocess_selected.py` | Figuras de ventana, filtrado y recorte |
| `PSIB.bat` | Preparación del entorno Windows, comprobación de dependencias y apertura del GUI |
| `scripts/check_requirements.py` | Comprueba las dependencias directas antes de iniciar el GUI |
| `scripts/gui.py` | Interfaz de escritorio para el flujo 2D, selección, análisis morfológico 3D, compresión y reconstrucción |
| `tests/` | Tests sintéticos con `unittest` |
| `data/` | Datos locales, ignorados por Git |
| `results/` | Tablas y figuras generadas, ignoradas por Git |

El paquete usa estructura `src/`. Los scripts agregan esa carpeta a `sys.path`;
no existe actualmente un archivo de empaquetado para instalarlo con `pip install .`.

## 3. Instalación y entorno

La versión objetivo es **Python 3.12**. Ejecutar desde la raíz del repositorio.

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si `.venv` ya existe, activarlo y volver a instalar los requisitos si cambiaron.
Comprobar el intérprete con
`python --version`. El entorno virtual puede coexistir con otra versión de
Python instalada en el sistema.

### Windows (CMD)

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

No copiar el entorno virtual entre sistemas operativos.

En Windows se puede ejecutar `PSIB.bat` con doble clic. Crea `.venv` si
falta y verifica las versiones de todos los requisitos directos. Solo ejecuta
`pip install -r requirements.txt` si falta alguna dependencia o su versión no
cumple la restricción; después comprueba consistencia con `pip check`. Luego
abre la GUI. El flujo Python `scripts/run_tpi.py` usa `data_dir`
con la precedencia habitual de `config.local.yaml`. Si no encuentra pares
CT/máscara, intenta `data/Selected_database/`; si tampoco encuentra pares allí,
termina con un error. No modifica los archivos YAML del usuario.

Instalar Python 3.12 de 64 bits con el lanzador `py` para crear el entorno.
La ejecución nativa del archivo BAT no se ha verificado desde el entorno Linux.

Para usar la interfaz gráfica, ejecutar `PSIB.bat`. La ventana ofrece
cinco secciones desplazables en una sidebar izquierda que se puede contraer con
el chevron o `Ctrl+B`; el visor ocupa el área derecha. La sección **Caso** permite
elegir un caso (se carga solo al seleccionarlo) y recorrer cortes, **Preprocesamiento** y **Segmentación**
actualizan el corte visible, **Cuantificación** ejecuta el análisis 3D de
referencia y **Resultados** lista figuras/CSV junto con sus explicaciones y el
registro. El pie fijo mantiene visibles los pasos, el progreso, Cancelar y la
acción principal. En la rail contraída, cada icono abre directamente su sección.
La barra de estado inferior presenta coordenadas y HU. El perfil se conserva en
`.psib_gui/`, ignorado por Git. El pipeline y la morfología corren en segundo
plano; los controles visibles se guardan en un YAML local ignorado por Git para
que el pipeline use la misma configuración.

La selección 2D y la segmentación base comparten la configuración resuelta.
El identificador usa la hora local con formato `YYYYMMDD-HHMMSS`; las salidas
van a `<output_dir>/segmentation/<identificador>/`. Si ya existe, esperar un
segundo y volver a ejecutar. Un fallo de instalación o de `pip check` detiene
el flujo antes de procesar datos.

Las dependencias base son NumPy, SciPy, scikit-image, NiBabel, Matplotlib,
OpenCV, pandas y PyYAML. OpenCV está declarado como dependencia, aunque los
módulos actuales no lo importan. Todos los flujos se ejecutan con Python
y las dependencias de `requirements.txt`.

Los requisitos expresan versiones mínimas, no versiones exactas. Para reproducir
experimentos, conservar además las versiones efectivamente instaladas.

## 4. Datos y convenciones espaciales

La configuración base apunta al subconjunto local `data/Selected_database/`,
con 10 pares CT/máscara. Los datos no se descargan automáticamente ni se
incluyen mediante Git. Conservar la información de licencia y procedencia
del README y `dataset.json` del dataset.

Se admiten dos convenciones de nombres, en las carpetas `imagesTr/` y
`labelsTr/`, o directamente en el directorio configurado:

```text
data/Selected_database/
  imagesTr/liver_3.nii.gz
  labelsTr/liver_3.nii.gz

data/Task03_Liver/
  volume-3.nii.gz
  segmentation-3.nii.gz
```

Se aceptan `.nii` y `.nii.gz`. En el formato `liver_N`, la máscara debe tener
el mismo nombre y extensión que la imagen. Solo se listan casos con ambos
archivos; los identificadores duplicados generan un error. Los archivos
auxiliares cuyo nombre empieza con `._` se ignoran. `imagesTs/` no participa.

### Etiquetas y ejes

| Valor | Significado |
| --- | --- |
| 0 | Fondo |
| 1 | Hígado sin etiqueta tumoral |
| 2 | Tumor |

- Región hepática completa: `seg > 0`, incluyendo el tumor.
- Referencia tumoral: `seg == 2`.
- Los arrays se reorientan a la orientación canónica más cercana a RAS mediante
  NiBabel; no se realiza un remuestreo a resolución isotrópica.
- Orden de ejes: `(X, Y, Z)`. Corte axial: `array[:, :, k]`, con índice desde 0.
- Espaciado: `(sx, sy, sz)` en milímetros, siguiendo los ejes del array.
- Las figuras muestran `image.T` con `origin="lower"`; esa transposición es de
  visualización y no cambia la convención de los arrays de procesamiento.

### API de carga

| Función | Resultado |
| --- | --- |
| `list_cases(data_dir)` | Lista ordenada de identificadores con pares disponibles |
| `load_case(case_id, data_dir=None)` | `(ct_hu, seg, spacing)`; CT `float32`, etiquetas `uint8` |
| `load_segmentation(case_id, data_dir=None)` | Solo las etiquetas canónicas, evitando cargar todo el CT |
| `get_slice(volume, k)` | Corte 2D de un volumen 3D; error si `k` está fuera de rango |
| `iter_case_slices(case_id, slice_indices=None, data_dir=None, *, mask_only=False)` | Iterador de `(k, ct_2d, seg_2d, spacing)` sin cargar volúmenes completos |

`load_case` comprueba forma, matriz afín canónica, espaciado e integridad de
etiquetas. La carga de intensidades aplica la escala e intercepto del NIfTI
mediante `get_fdata()`. Si se omite `data_dir`, se utiliza la configuración.

## 5. Configuración

Por defecto, los scripts leen `config.yaml` desde la raíz. Si existe un archivo hermano
`config.local.yaml`, se carga ese archivo **en reemplazo completo** del base:
no hay combinación recursiva. Para mantener los valores del proyecto al
personalizar rutas, copiar primero el YAML completo.

Las rutas relativas se resuelven desde la carpeta del archivo elegido.
`load_config(path=None)` devuelve un diccionario con rutas absolutas en
`data_dir` y `output_dir`, y agrega `_root` y `_output_dir`. Además, crea el
directorio de salida y su subdirectorio `figures/`.

Valores del YAML base:

| Parámetro | Valor | Uso |
| --- | --- | --- |
| `data_dir` | `data/Selected_database` | Pares de entrada |
| `output_dir` | `results` | Salidas |
| `selection.candidate_pool_size` | 15 | Máximo de candidatos muestreados |
| `selection.n_cases` | 1 | Casos seleccionados; reducido temporalmente para una prueba rápida |
| `selection.slices_per_case` | 5 | Cortes solicitados por caso; el script admite 3 a 5 |
| `selection.slice_strategy` | `stratified` | Roles tumorales o `area_top` legado |
| `selection.minimum_slice_gap_mm` | 10 | Separación física entre cortes estratificados |
| `selection.min_tumor_pixels` | 20 | Área mínima para roles de tumor pequeño/mediano |
| `selection.minimum_slice_gap` | 2 | Separación antigua en índices, solo para `area_top` |
| `selection.seed` | 2026 | Semilla del muestreo |
| `dataset_context.density_tolerance_hu` | 5 | Diferencia absoluta considerada isodensa |
| `features.glcm_levels` | 16 | Niveles de cuantización GLCM; 32 producía matrices ralas en ROIs chicas |
| `lesion_eval.min_iou` | 0.0 | IoU mínimo entre componentes para aceptar correspondencia; 0 exige intersección |
| `morphology3d_prediction.min_component_voxels` | 8 | Tamaño mínimo de componente 3D de predicción que se conserva |
| `compression.qualities` | `[95, 75, 50, 25, 10]` | Calidades JPEG simuladas/medidas |
| `reconstruction.cases` | `3` | Primeros tres casos seleccionados; admite lista explícita de IDs |
| `reconstruction.slices_per_case` | 3 | Roles por caso: máximo, pequeño y libre de tumor |
| `reconstruction.angles` | `[180, 60, 20]` | Número de proyecciones simuladas |
| `reconstruction.downsample` | 1 | Factor entero opcional de reducción para acelerar |
| `reconstruction.sart_iterations` | 3 | Iteraciones SART por corte y número de ángulos |
| `reconstruction.sart_relaxation` | 0.15 | Relajación de SART |
| `preprocessing.hu_window.center` | 60 | Centro HU |
| `preprocessing.hu_window.width` | 400 | Ancho HU |
| `preprocessing.denoising.method` | `median` | Filtro activo |
| `preprocessing.denoising.median_size` | 3 | Ventana mediana impar |
| `preprocessing.denoising.gaussian_sigma` | 1.0 | Sigma gaussiano en píxeles |
| `preprocessing.denoising.bilateral_sigma_color` | 0.08 | Escala de intensidad normalizada |
| `preprocessing.denoising.bilateral_sigma_spatial` | 3.0 | Escala espacial del bilateral |
| `preprocessing.enhancement.method` | `none` | Realce desactivado |
| `preprocessing.enhancement.clahe_clip_limit` | 0.01 | Límite de CLAHE |
| `preprocessing.enhancement.clahe_kernel_size` | 8 | Tamaño de región CLAHE |
| `preprocessing.crop_margin` | 8 | Margen del recorte en píxeles |

El bloque `segmentation` contiene los valores de la sección 8. Admiten
`--config`: `run_tpi.py`, `select_slices_2d.py`, `run_segmentation.py`,
`compare_filters.py`, `extract_features.py` y `evaluate_lesions.py`. `explore_dataset.py`, `select_cases.py` y
`preprocess_selected.py` usan el YAML de la raíz. Todos conservan la precedencia
completa de `config.local.yaml`.

`candidate_pool_size` se utiliza solo en el inventario volumétrico. La selección
2D usa `n_cases`, `slices_per_case`, `slice_strategy`, `minimum_slice_gap_mm`,
`min_tumor_pixels` y `seed` sin calcular terciles ni componentes 3D. Las claves
nuevas tienen valores por omisión en el código para que un `config.local.yaml`
anterior que reemplaza completamente al base siga funcionando. El fallback de datos pertenece únicamente a
`run_tpi.py`; los scripts individuales usan `data_dir` directamente.

## 6. Flujos ejecutables y archivos de salida

### Flujo principal: selección 2D y segmentación base

Con el entorno activado y los requisitos instalados:

```bash
python scripts/run_tpi.py
# Alternativa con configuración propia:
python scripts/run_tpi.py --config ruta/a/config.yaml
```

El lanzador busca pares en el `data_dir` efectivo. Si no hay pares, intenta
`data/Selected_database/` bajo la raíz del repositorio. Si tampoco hay pares
allí, falla antes de seleccionar. El fallback también se aplica a una carpeta
existente pero vacía; no corrige datos corruptos ni una selección incompatible
con la cantidad de casos disponibles.

Regenera `<output_dir>/selected_cases_slices.csv` y ejecuta segmentación base
con un ID de hora local como `20261031-180409`. Comprueba que la carpeta de
esa ejecución no exista antes de regenerar la selección. No ejecuta barridos,
comparación de filtros ni inventario 3D. Cada lanzamiento vuelve a seleccionar;
para reutilizar una selección, usar el ejecutor manual de la sección 14.

Ambas etapas reciben una configuración temporal con rutas absolutas y el mismo
`data_dir` resuelto. No se modifica el YAML del usuario. Para reproducir una
corrida, consultar `effective_config.yaml`: la ruta de configuración registrada
en `run.json` apunta a ese archivo temporal, eliminado al terminar el lanzador.
Los scripts individuales no aplican fallback; si se utilizó, corregir `data_dir`
antes de comparar filtros o ejecutar un barrido con la misma selección.

La llamada directa a Python no instala dependencias. En Windows,
`PSIB.bat` prepara y verifica el entorno antes de abrir la GUI.
Las salidas de segmentación se describen en la sección 14.

### Flujo opcional: exploración volumétrica

El flujo siguiente carga volúmenes completos y cuenta componentes 3D.
Reservarlo para una máquina con memoria suficiente:

```bash
python scripts/explore_dataset.py
python scripts/select_cases.py
python scripts/preprocess_selected.py
```

Ejecutar en ese orden. Si cambia el dataset o la selección, regenerar las
tablas antes de generar las figuras; no hay detección automática de tablas
desactualizadas. Las ejecuciones reutilizan los nombres de salida y pueden
sobrescribir archivos previos.

### Exploración

`explore_dataset.py` muestrea sin reemplazo hasta
`min(candidate_pool_size, casos_disponibles)` casos con la semilla indicada.
Con 10 casos locales y un máximo de 15, el inventario incluye los 10.

Genera:

- `results/candidate_pool.csv`: identificador, orden, semilla y criterio de muestreo.
- `results/dataset_inventory.csv`: dimensiones, espaciado, estadísticas HU,
  conteos de etiquetas, volumen tumoral, componentes y contexto HU/hepático.
- `results/dataset_context_table.csv`: tabla compacta para el informe, con caso,
  geometría, volumen, lesiones, burden, delta HU, densidad y categorías de tamaño.
- `results/figures/hu_histogram.png`.
- `results/figures/tumor_volume_distribution.png`.
- `results/figures/spacing_distribution.png`.
- `results/figures/delta_hu_vs_tumor_volume.png`: delta HU firmado frente al
  volumen tumoral, con cada caso rotulado.

El histograma HU usa hasta 100 000 vóxeles por caso del pool, muestreados en
todo el CT. El volumen tumoral en cm³ se calcula como
`tumor_voxels * sx * sy * sz / 1000`. La columna `liver_voxels` cuenta solo
la etiqueta 1 (hígado sano) y excluye tumor; `liver_total_voxels` cuenta
`seg > 0`, y `liver_volume_cm3` usa esa región total. `tumor_burden` es el
cociente entre vóxeles tumorales y `liver_total_voxels`. Las medianas HU se
calculan en `seg == 1` y `seg == 2`; `delta_hu` es tumor menos hígado sano y
conserva el signo. `contrast_class` usa magnitud absoluta en `<20`, `[20,40)`,
`[40,60)` y `>=60` HU. `density` clasifica el delta como hipodenso, hiperdenso
o isodenso dentro de `dataset_context.density_tolerance_hu` (5 HU por defecto).
Los conteos `lesions_lt10mm`, `lesions_10_20mm` y `lesions_gt20mm` agrupan
componentes 6-conexos por diámetro de esfera equivalente (10–20 incluye ambos
extremos). El conteo existente
El conteo `tumor_lesions_3d` usa componentes conectados con la conectividad
predeterminada de SciPy en 3D: 6 vecinos. Es un conteo de componentes, no una
identificación clínica de lesiones.

### Selección

`select_cases.py` requiere el inventario y genera
`results/selected_cases_slices.csv`.

Los casos se distribuyen por terciles de volumen tumoral (`small`, `medium`,
`large`) y se prioriza incluir hasta dos casos con múltiples componentes
tumorales cuando estén disponibles. Se informa un error si faltan tres
terciles distinguibles, se solicitan más casos que los disponibles o no hay
ningún caso con múltiples componentes.

La estrategia predeterminada asigna, en orden, los roles de tumor máximo,
mediano, pequeño, percentil 75 y libre de tumor; con 3 cortes usa máximo,
pequeño y libre de tumor, y con 4 agrega mediano. Los roles se separan por
`minimum_slice_gap_mm` usando el `sz` propio del caso. Los roles que no puedan
completarse se registran en `selection_criteria` y producen una advertencia;
la selección continúa. `area_top` conserva el ranking anterior y la distancia
por índices de `minimum_slice_gap` para reproducibilidad. El flujo 2D solo lee
planos de máscaras (`mask_only=True`), sin cargar CT ni materializar volúmenes.

El CSV incluye `selection_rank`, `case_id`, `slice_index`, `tumor_volume_cm3`,
`tumor_lesions_3d`, `burden_tercile`, `candidate_pool_size`, `seed` y
`selection_criteria`. Ambos selectores preservan sus columnas originales y
agregan `slice_role`, `tumor_area_px`, `liver_area_px` (etiquetas `seg > 0`) y
`slice_position_mm` (`slice_index * sz`). Lectores existentes deben ignorar las
columnas extra y conservar los nombres conocidos.

### Figuras de preprocesamiento

`preprocess_selected.py` carga cada caso una vez por grupo de cortes. Por cada
fila del CSV crea una figura con ventana HU, imagen filtrada y recorte:

```text
results/figures/preprocess_case_003_slice_042.png
```

Los índices del nombre son ejemplos. El script exporta figuras PNG, no arrays
de preprocesamiento ni máscaras tumorales.

### Experimento de compresión

Después de crear `selected_cases_slices.csv`:

```bash
python scripts/run_compression.py --run-id compresion-base
```

El programa procesa todos los pares caso/corte seleccionados y crea un
directorio exclusivo en `results/compression/<run-id>/`. La entrada es la salida
normalizada de `prepare_denoised_slice`, cuantizada como `round(255*x)`; ese
array de 8 bits es el control sin compresión. Ninguna etiqueta de tumor se
entrega a `segment_slice`: el algoritmo recibe solo la imagen y `seg > 0`; las
referencias `seg == 2` se usan después para métricas y para definir las ROIs de
textura.

Las condiciones sin pérdida son PNG OpenCV nivel 9 (con comprobación de
igualdad exacta al decodificar), estimación Huffman por entropía/código medio y
RLE fila por fila. Los tamaños Huffman y RLE no incluyen metadatos; Huffman
guarda también su tabla en `huffman_code_table_json`. JPEG simulado usa DCT 8×8,
tabla de luminancia estándar escalada por calidad IJG, orden zigzag, RLE y
longitudes Huffman estimadas. Se compara la estimación con el tamaño del archivo
JPEG real de OpenCV a la misma calidad. La imagen reconstruida por DCT, no la
decodificación de OpenCV, alimenta métricas, segmentación y textura.

Cada fila calcula razón `bits_original / bits_comprimidos`, redundancia relativa
`1 - 1/C`, bits por píxel, PSNR y SSIM frente al control, además de Dice e IoU.
La tabla incluye propiedades de textura de tumor y de hígado sano para cada
propiedad numérica que devuelva `extract_texture_features`, y su cambio relativo
frente al control. Si una propiedad base vale cero o no es calculable, el cambio
relativo es NaN.

| Salida | Contenido |
| --- | --- |
| `per_slice.csv` | Medidas por caso, corte y condición; código Huffman JSON y cambios de textura |
| `per_case.csv` | Medias de cortes por caso y condición |
| `summary.csv` | Conteos, medias y desvíos por condición/calidad |
| `effective_config.yaml`, `selected_cases_slices.csv`, `run.json` | Configuración, selección y estado/tiempos |
| `figures/compression_ratio_vs_quality.png` | Razón de compresión JPEG real contra calidad |
| `figures/dice_vs_quality.png` | Dice de tumor frente a calidad |
| `figures/texture_relative_change_vs_quality.png` | Cambio de cada descriptor textural frente a calidad |
| `figures/compression_example_grid.png` | Control, JPEG q75/q25/q10 y diferencias absolutas |

### Simulación de reconstrucción

LiTS contiene tomografías ya reconstruidas. Este flujo simula adquisición
tomográfica desde un corte; no representa un escáner real ni reconstruye datos
crudos de pacientes. Ejecutar luego de crear la selección:

```bash
python scripts/run_reconstruction.py --run-id reconstruccion-base
```

Por defecto toma los primeros tres casos seleccionados y, por caso, los roles
`largest`, `small` y `tumor_free`. Para selecciones antiguas sin roles, toma
hasta tres cortes del caso. Convierte HU a `mu=max(HU+1000,0)/1000`, extrae la
caja hepática con margen y la centra en una matriz cuadrada cuyo lado es al
menos la diagonal de la caja. Esto coloca toda la caja dentro del círculo
inscrito requerido por `radon(circle=True)` de scikit-image. Puede usarse
`reconstruction.downsample` para reducir el tamaño antes de proyectar; luego se
interpola la reconstrucción al tamaño de la caja original.

La transformación especificada limita HU menores que −1000 a `mu=0`, por lo
que esos valores no se pueden recuperar exactamente al volver a HU.

Se comparan suma de retroproyecciones, FBP con rampa, FBP con Hann y SART para
180, 60 y 20 ángulos por defecto. SART itera sobre el sinograma según
`reconstruction.sart_iterations` y `sart_relaxation`. Cada resultado se pega en
la imagen original, se pasa por la misma ventana/filtro y configuración de
segmentación, y se evalúa con Dice/IoU contra `seg == 2` (entrada hepática
`seg > 0`). PSNR/SSIM comparan el recorte de la caja en escala normalizada de
ventana HU. La estimación de tiempo se actualiza al procesar cortes y `run.json`
guarda el tiempo medido.

| Salida | Contenido |
| --- | --- |
| `per_slice.csv` | PSNR, SSIM, Dice, IoU y cambio de Dice por corte, método y ángulos |
| `per_case.csv` | Medias de cortes por caso/método/ángulos |
| `summary.csv` | Medias y desvíos entre casos |
| `effective_config.yaml`, `selected_cases_slices.csv`, `run.json` | Configuración, subconjunto y estado/tiempos |
| `figures/sinogram.png` | Proyecciones de ejemplo |
| `figures/reconstruction_grid.png` | Original y cuatro métodos para cada cantidad de ángulos |
| `figures/dice_vs_angles.png` | Dice medio frente al número de ángulos |

Los dos experimentos son descriptivos sobre cortes seleccionados con referencia;
no son validación independiente. La ejecución de reconstrucción puede ser lenta
en resolución original, en especial con SART. Las estimaciones Huffman/RLE y
JPEG codificado teóricas no equivalen a formatos autocontenidos con tablas,
cabeceras, byte-stuffing o marcadores de final de bloque. La configuración
`segmentation.stats_scope: case` requiere que el módulo y API `psib.case_stats`
se integren con `segment_slice`; en su ausencia los experimentos fallan con un
mensaje claro, en vez de usar silenciosamente estadísticas por corte.

## 7. API de preprocesamiento

`preprocess_slice(slice_hu, liver_mask, cfg)` acepta la configuración completa
o directamente el bloque de preprocesamiento y devuelve:

| Clave | Contenido |
| --- | --- |
| `windowed` | Imagen recortada a la ventana HU y normalizada en `[0,1]` |
| `denoised` | Imagen completa filtrada |
| `enhanced` | Imagen completa con realce opcional |
| `cropped` | Recorte de `enhanced` alrededor del hígado |
| `crop_box` | Límites `(x0, y0, x1, y1)`, con extremos superiores exclusivos |

La ventana actual cubre `[-140, 260] HU`. Se admiten filtros `median`,
`gaussian` y `bilateral`; el realce admite `none`, `histogram_equalization`
y `clahe`. El recorte agrega el margen configurado sin salir de la imagen.
Una máscara hepática vacía provoca `ValueError` porque no se puede definir
su caja de recorte.

Las funciones de ventana, filtros, realce y recorte también pueden usarse
por separado. Para reconstruir coordenadas desde un recorte, sumar su origen
`(x0, y0)`. Para la segmentación base se usa `denoised`, que conserva la
forma original y evita necesitar esa reconstrucción.

`prepare_denoised_slice(slice_hu, cfg)` aplica solo ventana HU y denoising a un
corte completo y devuelve la imagen normalizada. El ejecutor de segmentación y
`extract_features.py` comparten esta función para analizar la misma etapa de
preprocesamiento.

## 8. API de segmentación

```text
segment_slice(image, liver_mask, settings) -> np.ndarray
```

Recibe una imagen 2D real, finita y normalizada en `[0,1]`, una máscara
hepática binaria de igual forma y un diccionario de parámetros. Devuelve una
máscara booleana de la misma forma. No modifica entradas ni usa la referencia
tumoral.

| Parámetro | Predeterminado | Opciones o condición |
| --- | --- | --- |
| `method` | `global` | `global`, `otsu`, `statistical` |
| `threshold` | 0.45 | Número finito en `[0,1]` |
| `k` | 1.0 | Número finito no negativo |
| `direction` | `lower` | `lower`, `upper`; afecta solo al método estadístico |
| `opening` | `False` | Booleano |
| `closing` | `False` | Booleano |
| `fill_holes` | `False` | Booleano |
| `footprint` | `disk` | `disk`, `square` |
| `radius` | 1 | Entero positivo |
| `min_area` | 0 | Entero no negativo, en píxeles; 0 desactiva el filtro |

Un diccionario vacío utiliza todos los valores predeterminados. Las claves
desconocidas y los valores inválidos generan `ValueError`, incluso si el
parámetro no interviene en el método elegido.

### Umbrales

- **Global:** `image < threshold` dentro del hígado.
- **Otsu:** calcula el umbral únicamente con intensidades hepáticas y selecciona
  las inferiores a ese umbral.
- **Estadístico:** calcula media y desvío poblacional (`ddof=0`) en la región
  hepática. Selecciona `image < media - k*desvío` para `lower`, o
  `image > media + k*desvío` para `upper`.

Todas las comparaciones son estrictas: la igualdad con el umbral queda fuera.
Los valores iniciales no han sido ajustados mediante experimentos.

El módulo no recibe HU. Convertir un umbral HU con
`(HU - límite_inferior) / ancho` antes de llamar; con la ventana actual,
`40 HU` corresponde a `0.45`. La conversión solo produce un umbral válido
para esta API si queda en `[0,1]`; el módulo no lo recorta automáticamente.

### Posprocesamiento y casos especiales

Orden: umbralización → apertura → cierre → relleno de huecos → filtro de
componentes. Cada operación es opcional. Apertura y cierre comparten el
elemento estructurante: disco de radio `radius` o cuadrado de lado
`2*radius + 1`. Los bordes externos de la imagen se consideran fondo.

La máscara se restringe al hígado después de cada operación morfológica.
Los componentes y el relleno usan 8 vecinos. Se conservan componentes de
área mayor o igual a `min_area`. El tamaño está en píxeles, no en mm².

Tras validar entradas y parámetros, un hígado vacío produce una máscara
vacía. Las intensidades hepáticas constantes producen una máscara vacía
con Otsu y el método estadístico; el global sigue comparando con su umbral.

## 9. API de métricas

```text
dice_score(pred, reference) -> float
iou_score(pred, reference) -> float
```

Aceptan máscaras binarias 2D o 3D con formas idénticas. Se admiten booleanos
y valores numéricos exactamente 0/1. No realizan broadcasting, recorte ni
redimensionamiento. Rechazan valores no binarios y no finitos.

Sean `P` la predicción y `R` la referencia:

- Dice: `2 * |P ∩ R| / (|P| + |R|)`.
- IoU: `|P ∩ R| / |P ∪ R|`.
- Ambas vacías: `1.0` en ambas métricas.
- Solo una vacía: `0.0`.

Se cuenta el conjunto de elementos de toda la máscara sin promediar por corte
y sin agregar épsilon. Ambas funciones son puras y devuelven un `float` en
`[0,1]`. Si las formas difieren, lanzan `ValueError` con ambas formas en el
mensaje. Formas iguales no garantizan alineación espacial: esa correspondencia
debe conservarla quien llama a la función.

## 10. Ejemplo integrado de uso manual

Ejecutar este código desde la raíz con el entorno activado, en Python.
Requiere haber generado previamente el CSV de selección.
Procesa un único corte y muestra sus métricas; no guarda archivos.

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd() / "src"))

from psib.config import load_config
from psib.data_io import iter_case_slices
from psib.preprocessing import preprocess_slice
from psib.segmentation import segment_slice
from psib.metrics import dice_score, iou_score

cfg = load_config(Path.cwd() / "config.yaml")
selected = pd.read_csv(Path(cfg["_output_dir"]) / "selected_cases_slices.csv")
row = selected.iloc[0]
k = int(row.slice_index)
_, slice_hu, labels, spacing = next(iter_case_slices(
    int(row.case_id), [k], cfg["data_dir"]))
liver = labels > 0
reference = labels == 2
processed = preprocess_slice(slice_hu, liver, cfg)

settings = {"method": "statistical", "k": 1.0, "direction": "lower"}
prediction = segment_slice(processed["denoised"], liver, settings)
print({
    "case_id": int(row.case_id),
    "slice_index": k,
    "dice": dice_score(prediction, reference),
    "iou": iou_score(prediction, reference),
})
```

Si el corte no contiene hígado, no puede usarse `preprocess_slice` porque
intenta recortarlo. El ejecutor utiliza directamente ventana y filtrado, sin recorte, y puede
procesar esos cortes con predicción vacía.

## 11. Comparación de filtros y tests

La exploración y selección se ejecutan con `explore_dataset.py` y
`select_cases.py`; `preprocess_selected.py` genera las vistas de ventana HU,
filtrado y recorte. Para una selección con lectura por cortes, usar
`select_slices_2d.py`.

Después de generar `selected_cases_slices.csv`, comparar los filtros con:

```bash
python scripts/compare_filters.py
python scripts/compare_filters.py --config /ruta/a/config.yaml
```

El script toma el primer par caso/corte del CSV y lee únicamente ese corte 2D.
Respeta la precedencia de `config.local.yaml`, la ventana HU y los parámetros
configurados de mediana, gaussiano y bilateral. Guarda cuatro paneles con la
misma escala `[0, 1]` y orientación en
`figures/filter_comparison_case_<id>_slice_<index>.png`, bajo el directorio de
salida configurado (identificadores con ceros a la izquierda). No abre una
ventana gráfica. Compara los tres filtros aunque `denoising.method` elija
uno distinto para la segmentación. Si falta el CSV o está vacío, informa un error.

Registrar la ventana y los parámetros elegidos; comparar reducción de ruido y
visibilidad de los bordes hepáticos y tumorales. La comparación visual no es
una métrica de precisión de segmentación. Los recortes de preprocesamiento
usan la máscara hepática de referencia; su caja `(x0, y0, x1, y1)` permite
sumar el origen a coordenadas locales para volver al corte completo. La
selección usa las máscaras de referencia y mantiene ese sesgo de muestreo.

Validar con datos sintéticos:

```bash
python -m unittest
python -m unittest tests.test_metrics
python -m unittest tests.test_segmentation
python -m pip check
```

| Archivo de tests | Cobertura principal |
| --- | --- |
| `test_data_io.py` | Pares NIfTI sintéticos, orientación, espaciado y caso faltante |
| `test_dataset.py` | Muestreo, contexto HU, tamaños físicos y selección estratificada |
| `test_preprocessing.py` | Ventana, coordenadas de recorte, forma y dtype de filtros |
| `test_compare_filters.py` | Parámetros de filtros, orientación, lectura de un corte y errores de selección |
| `test_metrics.py` | Casos idénticos, disjuntos, vacíos, solapamiento y validación |
| `test_segmentation.py` | Umbrales, región hepática, morfología, conectividad y contrato |
| `test_run_segmentation.py` | Ejecución, exportación, agregación, barrido, fallos y reproducibilidad |
| `test_compression.py` | Huffman, PNG reversible, DCT de calidad alta y tamaño JPEG |
| `test_reconstruction.py` | Conversión HU/mu, sinograma, FBP, SART y PSNR idéntico |
| `test_run_tpi.py` | Directorio configurado, fallback, precedencia local, ID temporal y ejecución integrada |
| `test_slice_io.py` | Lectura por planos, orientación, rango de etiquetas y selección 2D |
| `test_features.py` | Geometría 2D, espaciado físico y GLCM enmascarada |
| `test_morphology3d.py` | Medidas físicas de componentes y diámetros 3D |
| `test_morphology3d_prediction.py` | Limpieza de componentes y apilado volumétrico sintético |
| `test_lesion_eval.py` | Correspondencia por lesión, divisiones, fusiones y tamaños |
| `test_agreement.py` | Bland–Altman, Wilcoxon exacto y ajuste Holm |

La suite incluye tests sintéticos del lanzador y del ejecutor sin requerir
datos reales. No sustituye una evaluación de calidad de segmentación ni una
prueba de todos los scripts sobre los volúmenes reales. El test del lanzador
ejecuta el flujo Python; no prueba CMD ni el archivo BAT nativo de Windows.

## 12. Errores habituales y límites de la revisión

| Situación | Causa o acción |
| --- | --- |
| `ModuleNotFoundError: psib` al usar la API | Agregar `src/` al path, como en el ejemplo; los scripts ya lo hacen |
| No se encuentra un par CT/máscara | Revisar `data_dir`, nombres y existencia de ambos archivos |
| Formas, afines o espaciados incompatibles | Comprobar que CT y máscara pertenezcan al mismo caso y geometría |
| Falta `dataset_inventory.csv` | Ejecutar primero `explore_dataset.py` |
| Falta `selected_cases_slices.csv` | Ejecutar la selección antes del preprocesamiento |
| No se pueden formar los terciles | Revisar pool y empates en volúmenes tumorales |
| No hay casos multilesión | La selección volumétrica exige al menos uno; la selección 2D no |
| Fallo de instalación o `pip check` en el BAT | Revisar el entorno y el mensaje antes de procesar datos |
| El lanzador no encuentra pares en ninguna ruta | Restaurar el dataset o corregir `data_dir` |
| ID temporal ya existente | Esperar un segundo y volver a ejecutar el lanzador |
| La comparación de filtros falla tras un fallback | Ajustar `data_dir`; el fallback no modifica el YAML |
| No se puede recortar una máscara vacía | Revisar el corte elegido o tratarlo antes del recorte |
| Valores HU enviados a segmentación | Usar la imagen normalizada `denoised` |
| Formas distintas en segmentación o métricas | Conservar las coordenadas completas o reconstruir el recorte; no redimensionar para ocultar el desajuste |

La revisión incluyó los módulos, scripts, configuración y tests.
La revisión se complementa con la suite sintética. Los archivos de ejecuciones
locales se guardan en `results/`, fuera de Git.

Detalles a considerar en trabajos futuros:

- La validación de etiquetas de los cargadores volumétricos originales convierte
  a `uint8` antes de comprobar
  el conjunto `{0,1,2}`; no debe asumirse como una validación exhaustiva de
  etiquetas arbitrarias fuera de rango. El lector nuevo por cortes valida el
  rango antes de convertir a `uint8`.
- El README describe el tamaño del conjunto original de LiTS. El script de
  exploración informa la cantidad de casos pareados disponibles en la ruta local.
- El lanzador y la comparación de filtros prueban la precedencia de configuración
  local. No hay cobertura de todas las validaciones de NIfTI; las garantías de
  tests son las indicadas arriba.

## 13. Interpretación experimental y próximos pasos

El objetivo mínimo y máximo se definen en la sección 1. La segmentación y las
características se calculan en cortes 2D; el componente 3D del objetivo máximo
se limita a volumen y conteo de componentes de las máscaras de referencia.
La referencia tumoral se usa para seleccionar cortes y evaluar, pero no se
entrega al algoritmo de segmentación.

El ajuste previsto sobre los mismos 10 casos será exploratorio. Una métrica
por caso sobre esos cortes no equivale a evaluar el volumen completo. La
comparación usa Dice e IoU para más de una familia de umbralización (global,
Otsu y estadística), con variantes morfológicas. La evaluación posterior en
una base mayor deberá excluir los casos usados en el ajuste y conservar la
configuración elegida para esa evaluación.

El ejecutor y el barrido permiten la comparación exploratoria sobre los casos
locales. Queda completar y analizar el barrido; la validación independiente en
otra base se considera trabajo posterior y queda fuera de los resultados
actuales.

## 14. Ejecutor de segmentación y barrido

Desde la raíz, usando exclusivamente procesamiento 2D:

```bash
python scripts/select_slices_2d.py
python scripts/run_segmentation.py --run-id base
python scripts/run_segmentation.py --run-id comparacion --sweep
python scripts/run_segmentation.py --config config.yaml --run-id alternativa
```

`--run-id` es obligatorio y admite letras ASCII, números, guiones y guiones bajos.
La carpeta `<output_dir>/segmentation/<run-id>/` debe ser nueva, incluso si una
corrida anterior falló. Este ejecutor manual no regenera la selección automáticamente; `run_tpi.py` sí. Para otro
archivo de configuración, los CSV de entrada deben estar en su directorio de
salida efectivo.

Sin `--sweep` se usa el bloque YAML `segmentation`. En barrido se reemplaza ese
bloque por una grilla fija, conservando el preprocesamiento:

- Umbrales globales 30, 40 y 50 HU; Otsu; estadístico inferior con k=0.5, 1 y 1.5.
- Seis perfiles: ninguno; apertura disco radio 1; cierre disco radio 1;
  apertura+cierre disco radio 1; apertura+cierre disco radio 2;
  apertura+cierre cuadrado de lado 3.
- Relleno de huecos desactivado/activado y área mínima 0, 25 o 100 píxeles.

El producto genera 252 configuraciones con identificadores estables `cfg_000`
a `cfg_251`. Los umbrales HU se convierten usando la ventana efectiva; si alguno
queda fuera de ella, la corrida falla. No se ajusta el preprocesamiento en la grilla.

Cada caso se abre una vez en la primera pasada y solo se leen los planos elegidos;
cada corte se filtra una vez. No se construye un array volumétrico completo.
Se usa el mismo resultado `denoised` del preprocesamiento, sin realce ni recorte.
Los cortes sin hígado se procesan con máscara vacía. En barrido se realiza una
segunda pasada por casos para exportar las tres variantes ganadoras.

### Archivos por ejecución

| Archivo | Contenido |
| --- | --- |
| `selected_cases_slices.csv` | Copia exacta de la selección de entrada |
| `effective_config.yaml` | Configuración efectiva con valores de segmentación completos, sin claves internas |
| `configurations.csv` | Identificador, parámetros y umbral HU cuando aplica |
| `metrics_per_slice.csv` | Configuración, caso, corte, áreas, intersección, unión, Dice e IoU |
| `metrics_per_case.csv` | Conteos acumulados sobre cortes seleccionados y métricas por caso |
| `summary.csv` | Cantidad de casos, media y desvío poblacional entre casos |
| `best_configurations.csv` | Mejor configuración de cada método; solo en barrido |
| `run.json` | Estado, argumentos, tiempos UTC, versiones, revisión Git y cambios locales |
| `predictions/<configuration_id>/*.npy` | Máscaras booleanas completas, sin transposición |
| `predictions/<configuration_id>/*.png` | Grilla cuadrada 2×2: imagen original con ventana HU, imagen denoised, referencia tumoral y predicción, con métricas |

Los CSV de métricas identifican la configuración mediante `configuration_id`;
los parámetros completos se consultan en `configurations.csv`. `pred_area` y
`reference_area` son conteos de píxeles, no superficies físicas.

Para cada caso se suman las áreas, intersecciones y uniones de sus cortes y
se calculan Dice e IoU desde esos totales. No se promedian las métricas por
corte. El resumen sí promedia las métricas entre casos, dando el mismo peso
a cada caso. Los desvíos son poblacionales (`ddof=0`).

Las ganadoras se eligen por Dice medio descendente, luego IoU medio descendente
y, ante empate, identificador ascendente. Es una configuración común para todos
los casos por método. Se guardan todas las métricas del barrido, pero solo máscaras
y figuras de las ganadoras. En modo simple se exportan todos los cortes de su
única configuración.

El estado pasa de `running` a `completed` solo al terminar tablas, máscaras y
figuras. Un error deja `failed`, mensaje y contexto de caso/corte/configuración
cuando corresponda. Los resultados parciales de una corrida fallida no deben
interpretarse como un estudio completo. Si el proceso se interrumpe externamente,
puede quedar en `running`; no hay reanudación automática.


### Lectura de cortes con memoria acotada

`iter_case_slices(case_id, slice_indices=None, data_dir=None, mask_only=False)`
produce `(k, ct_2d, seg_2d, spacing)` por corte. Comprueba la geometría desde
las cabeceras y lee un plano mediante el proxy NIfTI. Las inversiones y
permutaciones a orientación canónica se aplican sobre ese plano 2D. No llama
a `load_case` ni materializa arrays 3D. Con `mask_only=True`, CT es `None` y
solo se leen etiquetas. La descompresión de `.nii.gz` puede requerir leer datos
anteriores al corte, pero no almacenarlos como un volumen completo.

`select_slices_2d.py --config config.yaml` usa ese lector para recorrer máscaras
corte a corte y guardar únicamente sus conteos de área. Selecciona `n_cases`
con semilla fija (los 10 si se solicitan todos los disponibles), luego prioriza
área tumoral, área hepática e índice con la separación mínima existente.
Genera el mismo CSV de entrada que necesita el ejecutor, con columnas mínimas
más semilla y criterio, sin volumen tumoral, terciles ni componentes 3D.
Esta variante de selección se registra explícitamente en el CSV y en las notas.

Por la restricción de recursos se verifican las pruebas del lector y del flujo
2D, sin volver a ejecutar el inventario volumétrico sobre datos reales:

```bash
python -m unittest tests.test_slice_io tests.test_run_tpi tests.test_compare_filters
python -m unittest tests.test_run_segmentation tests.test_segmentation tests.test_metrics tests.test_preprocessing
```

Los NIfTI de prueba son fixtures sintéticos pequeños; no se realiza segmentación
ni análisis de componentes 3D de los volúmenes reales.


## 15. Visualización de contornos

`psib.visualization` ofrece helpers opcionales para comparar máscaras alineadas:

- `plot_overlay(image, prediction, reference, *, ax=None, title=None)` devuelve
  `(figure, axes)`; puede dibujar sobre ejes existentes.
- `save_overlay(image, prediction, reference, output_path, *, title=None, dpi=150)`
  crea las carpetas de destino, guarda la figura, la cierra y devuelve un `Path`.

La imagen debe ser 2D, real y finita; ambas máscaras deben tener la misma forma
y valores binarios 0/1. Los arrays siguen la convención `(X, Y)` y se transponen
solo al dibujar. La predicción aparece en cian y la referencia en rojo anaranjado.
`dpi` debe ser un entero positivo. Las entradas inválidas generan `ValueError`.

Estos helpers no se llaman automáticamente desde `run_segmentation.py`;
las figuras del ejecutor mantienen sus tres paneles separados. Para guardar
una superposición del ejemplo de la sección 10:

```python
from psib.visualization import save_overlay

save_overlay(
    processed["denoised"], prediction, reference,
    Path(cfg["_output_dir"]) / "figures" / "contours.png",
    title=f"Caso {int(row.case_id)} — corte {k}",
)
```

## 16. API de características geométricas y textura 2D

El módulo `psib.features` extrae propiedades de máscaras en un corte axial. No
calcula volumen 3D ni asocia automáticamente una lesión entre cortes.

### Geometría por lesión

`extract_geometric_features(tumor_mask, spacing)` recibe una máscara binaria 2D
y el espaciado `(sx, sy)` en milímetros, siguiendo los ejes del array. Separa
componentes conectados con conectividad de 8 y devuelve una fila por componente:

| Columna | Significado |
| --- | --- |
| `component_id` | Etiqueta del componente en el corte |
| `area_pixels` | Cantidad de píxeles de la lesión |
| `area_mm2` | Superficie usando el área física de cada píxel |
| `eccentricity` | Excentricidad de la elipse equivalente, entre 0 y 1 |
| `solidity` | Cociente entre el área y el área convexa |
| `max_diameter_mm` | Diámetro máximo de Feret en milímetros |
| `centroid_x_mm`, `centroid_y_mm` | Centroide en coordenadas físicas del corte |

El espaciado axial `sz` no interviene en medidas de una sola sección. Una
máscara vacía devuelve una tabla sin filas y con las mismas columnas.

### Textura GLCM en una región

`extract_texture_features(image, roi_mask, levels=16, distances=(1,), angles=...)`
recibe una imagen 2D normalizada a `[0, 1]` y una máscara binaria de la región
de interés con la misma forma. La función cuantiza intensidades en 16 niveles
por defecto y calcula matrices simétricas para cuatro direcciones en el plano
(0°, 45°, 90° y 135°) a distancia de un píxel. El default cambió de 32 a 16
porque las matrices de 32 niveles son ralas en lesiones chicas. Distancias,
niveles y ángulos pueden configurarse.

Solo cuenta pares donde ambos píxeles caen dentro de la región. Devuelve
contraste, homogeneidad, energía, correlación, entropía y ASM, promediados entre
desplazamientos con pares válidos. La energía es `sqrt(ASM)`; la entropía se
calcula como `-sum(p * log2(p))` sobre las entradas no nulas. Homogeneidad
corresponde al momento inverso de diferencia usado en la cátedra. También
devuelve `pair_count` y `n_offsets`. Si ningún desplazamiento contiene pares,
las seis propiedades son NaN y ambos conteos son cero. Para comparar textura de
las referencias, usar por separado `seg == 2` para tumor y `seg == 1` para
hígado no tumoral, junto con la imagen filtrada normalizada.

Estas características resumen una región en un corte; sus comparaciones son
descriptivas cuando la región se define con máscaras de referencia. La API
permite analizar tanto una referencia como una predicción, pero esta entrega no
debe interpretarse como evaluación independiente. El resumen generado es
descriptivo. `extract_features.py` además calcula pruebas exactas de signos
emparejadas por caso para cinco propiedades y ajusta los valores p con Holm.

### Tablas por selección

Desde la raíz del repositorio y después de generar `<output_dir>/selected_cases_slices.csv`:

```bash
python scripts/extract_features.py
python scripts/plot_features.py
python scripts/extract_features.py --levels 8 16 32
```

El script lee cada corte seleccionado con `iter_case_slices`, prepara la imagen
normalizada mediante `prepare_denoised_slice` y escribe estos archivos en
`<output_dir>/features/`:

| Archivo | Contenido |
| --- | --- |
| `texture_per_region.csv` | Una fila por corte y ROI (`tumor_reference` o `healthy_liver_reference`), con área, conteo de píxeles y propiedades GLCM |
| `texture_summary.csv` | Media y desvío estándar poblacional por región y propiedad, con cantidad de cortes válidos |
| `texture_per_case.csv` | Una fila por caso y región, con medias de los cortes seleccionados y número de cortes válidos |
| `texture_case_differences.csv` | Una fila por caso y propiedad, con textura tumoral menos textura de hígado sano |
| `texture_difference_summary.csv` | Media, mediana, desvío poblacional, dirección, prueba exacta de signos y p ajustado por Holm para las diferencias entre casos emparejados |
| `tumor_geometry.csv` | Una fila por componente tumoral 8-conectado, con características geométricas y ubicación de caso/corte |
| `diagnostics.txt` | Casos sin textura tumoral computable y motivo por corte |

`texture_per_region.csv` agrega `nan_reason`: `empty_roi`, `no_valid_pairs` o
vacío cuando hay valores computables. `pair_count` cuenta los pares espaciales válidos antes de simetrizar cada GLCM;
`n_offsets` cuenta las combinaciones distancia/ángulo que aportaron pares. Las
propiedades sin pares válidos son NaN. En la tabla de resumen, `n_slices` ignora
esos valores NaN para cada propiedad. Las máscaras de referencia definen ambas
regiones, por lo que la comparación de tejidos es descriptiva. Para las
diferencias por caso, la prueba exacta bilateral de signos evalúa si hay una
dirección consistente (los empates se excluyen del contraste), y Holm corrige
cinco comparaciones: contraste, homogeneidad, energía, correlación y entropía.
ASM se informa, pero no se prueba porque es redundante con energía. Con
`--levels 8 16 32`, cada nivel guarda sus tablas en `features/levels_<n>/` y
`texture_levels_sensitivity.csv` resume las medianas por región y propiedad.
Es un análisis exploratorio con pocos casos seleccionados, no una validación
independiente de la segmentación.

El segundo comando lee ambas tablas de textura y guarda dos figuras, cada una
con cinco paneles: contraste, homogeneidad, energía, correlación y entropía. En la figura
por corte, cada caja resume los cortes con valores finitos; en la figura por
caso, resume las medias de cortes seleccionados dentro de cada caso. La línea es
la mediana, el triángulo es la media y el tamaño muestral aparece en el eje
horizontal. Los gráficos son descriptivos y no incluyen pruebas de significancia.

`figures/glcm_tissue_comparison.png` compara cortes. `figures/glcm_tissue_comparison_by_case.png`
compara medias de los cortes seleccionados dentro de cada caso y conecta con
líneas las dos regiones medidas en el mismo caso. En los datos actuales hay 9
pares completos: un caso no aporta textura tumoral calculable. Las cajas por
caso muestran variación entre casos, pero no reemplazan una evaluación sobre
volúmenes completos ni una prueba de significancia.

Las dos tablas de diferencias incluyen solo casos con valores calculables para
ambas regiones. La diferencia se define como tumor menos hígado sano; positiva
indica que el descriptor fue mayor en tumor para ese caso. La tabla resumen
describe magnitud y dirección entre los pares disponibles. Las pruebas tienen
poca potencia con el tamaño de muestra seleccionado y no demuestran separación
generalizable.

El comando acepta `--config ruta/a/config.yaml`. Usa el `data_dir` configurado
sin aplicar el fallback del lanzador; corregir esa ruta si fue necesario el fallback.

### Evaluación por lesión, concordancia y comparación de métodos

Tras completar una corrida de segmentación, `evaluate_lesions.py` evalúa cada
corte seleccionado contra la referencia. Por defecto elige la configuración
con mejor Dice medio, conserva la corrida y crea `<run>/lesion_eval/`; falla
si esa carpeta ya existe. La selección debe estar copiada dentro de la corrida
y las predicciones guardadas por el ejecutor:

```bash
python scripts/evaluate_lesions.py --run-id <run-id>
python scripts/evaluate_lesions.py --run-id <run-id> --configuration-id cfg_0001
```

La correspondencia usa componentes 8-conectados. Una referencia dividida entre
predicciones solapadas se agrupa; referencias que comparten una predicción se
consideran fusionadas. `lesion_eval.min_iou` controla el solapamiento mínimo
por enlace; el default 0 exige intersección positiva. Las clases `<10`,
`10-20` y `>=20` usan el Feret máximo físico de referencia.
`lesion_matches.csv` contiene TP/FP/FN, áreas físicas, error relativo firmado
de área, errores de Feret y Dice por grupo. `detection_per_case.csv` resume
TP/FP/FN, precisión, recall y F1 sobre los cortes seleccionados de cada caso;
`detection_summary.csv` agrega medias por caso y recall por tamaño. Es una
evaluación 2D sobre los cortes seleccionados, no seguimiento entre cortes ni
validación volumétrica.

El análisis de concordancia requiere la tabla anterior:

```bash
python scripts/analyze_agreement.py --lesion-matches results/segmentation/<run-id>/lesion_eval/lesion_matches.csv
```

Guarda diferencias absolutas y porcentuales, sesgo, límites de concordancia al
95 %, Pearson y Spearman para área y Feret máximo, además de figuras
Bland–Altman con puntos coloreados por clase. Solo incluye grupos TP.

La comparación de dos configuraciones usa Dice por caso, no los cortes como
observaciones independientes:

```bash
python scripts/compare_methods.py --run-id <run-id> --compare best:global,best:otsu
```

Se pueden repetir opciones `--compare`; `best:<method>` selecciona el mejor
Dice medio del método y también se aceptan IDs `cfg_xxxx`. Cada comparación
aplica Wilcoxon bilateral exacto y el conjunto de valores p se ajusta con Holm.
El CSV advierte cuando hay menos de seis casos emparejados. La salida se crea
en una subcarpeta de `method_comparisons/`.

### Morfología de referencia 3D

`analyze_morphology_3d.py` carga los volúmenes de etiquetas completos de los
casos seleccionados, identifica componentes tumorales 6-conectados y exporta
superficies. El CSV por lesión conserva volumen y extensiones de caja y agrega
Feret máximo 3D, máximo Feret axial a través de los cortes, diámetro de esfera
equivalente y diagonal de caja. Feret 3D usa distancias entre puntos extremos
de la envolvente convexa; Feret axial es el mayor Feret 2D de cualquier corte.
El CSV por caso informa volumen hepático (`seg > 0`) y carga tumoral (volumen
tumoral / volumen hepático). La GUI muestra Feret axial para la mayor lesión;
el CSV conserva diagonal de caja como otra medida. Es cuantificación de las
máscaras de referencia, no segmentación automática 3D.

### Estimación 3D apilando predicciones 2D

Este flujo opcional extiende una corrida 2D completada: aplica a cada corte
axial de todos los casos seleccionados la configuración con mejor Dice medio
por caso (o una configuración indicada), apila las predicciones, elimina
componentes 6-conectados menores que el umbral configurado y las compara con
las etiquetas tumorales de referencia. La región de búsqueda continúa usando
`seg > 0` como máscara hepática de referencia, igual que el flujo 2D; por eso
esta medida no equivale a segmentar automáticamente hígado y tumor sin ayuda.

Ejecutar desde la raíz del repositorio después de una corrida terminada:

```bash
python scripts/evaluate_morphology_3d.py --run-id <run-id>
python scripts/evaluate_morphology_3d.py --run-id <run-id> --configuration-id cfg_000
```

El default elimina componentes con menos de
`morphology3d_prediction.min_component_voxels` (8 vóxeles); `--min-component-voxels`
permite cambiarlo. Se leen todos los cortes completos de cada caso incluido en
la selección copiada a la corrida. El volumen se procesa de a un corte, pero
las máscaras 3D y el análisis de componentes requieren memoria adicional. El
tiempo puede ser considerable, en especial en NIfTI comprimidos. La salida se
crea en `<run>/morphology3d_prediction/` y nunca reemplaza una salida anterior.

| Archivo | Contenido |
| --- | --- |
| `morphology_3d_prediction_per_case.csv` | Volúmenes, Dice/IoU 3D, conteo de componentes y error de Feret axial por caso |
| `morphology_3d_prediction_per_lesion.csv` | Medidas geométricas de lesiones de referencia, predicción cruda y predicción filtrada |
| `comparison_3d.png` | Dispersión de volumen y Feret axial predichos frente a referencia |
| `case_NNN_masks_3d.npz` | Máscara de referencia, predicción cruda y predicción limpia del caso |
| `run.json` | Estado, configuración elegida y umbral de filtrado |

El caso mide Feret axial como el mayor diámetro de cualquier componente en
cualquier corte; el volumen usa el espaciado físico NIfTI. Dice e IoU se
calculan entre volúmenes binarios completos. El filtrado puede quitar falsos
positivos, pero también lesiones pequeñas; comparar `prediction_raw` y
`prediction_clean` para interpretar el efecto del umbral. Este apilado no añade
continuidad 3D al clasificador ni sustituye la validación independiente.
En la corrida base probada, el Dice 3D limpio medio fue 0.161 y varios casos
mostraron fuerte sobreestimación. El flujo funciona, pero antes de interpretar
la estimación 3D como resultado del TPI hay que mejorar y validar el
segmentador 2D.

## Interfaz GUI: distribución y controles

La interfaz Tkinter vive en `scripts/gui.py`; el estado de navegación de la
barra lateral y su persistencia se implementan sin Tk en `src/psib/gui_model.py`.
El ancho inicial es 340 px, el ancho expandido se limita a 300 px como mínimo
y al 45% del ancho de ventana como máximo. La ventana mínima es 1000 × 640.

```text
Ventana
├── Barra lateral izquierda (PANEL)
│   ├── Encabezado PSIB · TPI + chevron
│   ├── Navegación: Caso | Preprocesamiento | Segmentación | Cuantificación | Resultados
│   ├── Una sección visible; cada sección tiene su propio canvas vertical
│   └── Pie fijo: pasos, progreso, Cancelar y Ejecutar flujo 2D (acción principal)
├── Visor derecho (VIEW)
│   ├── Anterior/siguiente, zoom −/+, Ajustar, 100% y modo Overlay/Antes/Después
│   └── Canvas de imagen con desplazamiento por arrastre
└── Estado inferior a lo ancho: coordenadas de píxel, HU y estado del proceso
```

La sección **Caso** ofrece el selector de casos incluidos en
`selected_cases_slices.csv` (o todos los casos pareados antes de generar esa
selección), un selector de volumen LiTS, metadatos del caso, navegación axial y
saltos a cortes seleccionados. **Preprocesamiento** edita ventana HU, filtro,
parámetro, ecualización y recorte visual. **Segmentación** permite ajustar
umbralización, morfología, área mínima y overlays; al segmentar el corte actual
muestra Dice, IoU, área y componentes 2D en una tabla compacta. **Cuantificación**
ejecuta el análisis 3D existente sobre máscaras de referencia y permite iniciar
los experimentos de compresión y reconstrucción simulada. En **Caso**, el botón
**Generar selección de cortes 2D** crea `selected_cases_slices.csv` con la
configuración actual; esa selección es requisito para los dos experimentos. Si
falta, la GUI lo informa y permite generarla desde esa sección o ejecutar el
flujo 2D. Cada experimento guarda sus salidas en una carpeta con identificador
temporal bajo `results/compression/` o `results/reconstruction/`. La salida en
vivo y la cancelación se gestionan desde la interfaz; las figuras y tablas
exportadas se cargan en el visor compartido. **Resultados**
muestra las imágenes y CSV disponibles, su explicación (si existe una entrada
en `figure_notes.yaml`) y el registro del flujo.

La tabla 3D muestra volumen, componentes y Feret axial de la mayor lesión.
El análisis exporta además medidas 3D, extensiones por eje, carga tumoral y la
diagonal de caja en CSV.

Colapsar con el chevron o `Ctrl+B` cambia a una barra de iconos de 44 px. Los
iconos conservan la sección activa y al pulsar otro se expande esa sección. Los
widgets no se reconstruyen, por lo que se mantienen sus valores y posiciones de
scroll. El cambio vuelve a ajustar automáticamente el visor en modo Fit; en
zoom manual conserva el punto aproximado que estaba centrado. Las tareas se
ejecutan en segundo plano y muestran un indicador en la barra cuando está
colapsada. `ProfileStore` guarda `collapsed`, `active_section` y `width` en
`.psib_gui/profile.json`, ignorado por Git.

| Zona | Widgets principales | Acción |
| --- | --- | --- |
| Encabezado | nombre, chevron | Alternar sidebar; `Ctrl+B` hace lo mismo |
| Navegación | cinco botones / iconos del rail | Cambiar sección conservando su estado |
| Caso | selector, archivo, slider, spinbox | Cargar caso y navegar cortes |
| Preprocesamiento | entradas, combos, checkbox | Aplicar parámetros al corte visible |
| Segmentación | parámetros, toggles, Treeview | Segmentar corte y revisar métricas por lesión |
| Cuantificación | botón ejecutar/cancelar, Treeview | Procesar morfología de referencia en 3D |
| Resultados | Treeview, nota, abrir carpeta, log | Mostrar figura o vista previa de CSV en el visor |
| Visor | canvas, toolbar, navegación | Ver casos y resultados; zoom y paneo |

La GUI no implementa una predicción de segmentación 3D ni métricas automáticas
versus referencia 3D. El panel 3D presenta la morfología calculada desde las
etiquetas de referencia. Los controles de segmento actual son una vista previa;
el botón de ejecución usa los parámetros visibles y genera los resultados 2D
reproducibles del pipeline.

### Iconos de aplicación

El diseño fuente suministrado está en `assets/icon/liver_tumor_app_icon.png`.
`python scripts/build_icons.py` lo reconoce y crea la copia canónica
`assets/icon/psib_icon_master.png` junto con PNG de 16, 24,
32, 48, 64, 128, 256 y 512 px, el multi-size `assets/icon/psib.ico` y
`assets/logo.png` de 56 px (2× el icono de 28 px del encabezado). El script usa
Pillow, LANCZOS, no cambia el master y puede repetirse. No genera figuras nuevas
ni redibuja el diseño. Tras añadir el master, revisar visualmente los PNG de 16
y 32 px, en particular que siga distinguiéndose el círculo tumoral.

`src/psib/app_icon.py` calcula las rutas y carga las imágenes después de crear
Tk; conserva las referencias `PhotoImage` en la instancia de la GUI. Si falta
cualquier PNG, el ICO, el logo, o falla la lectura, el cargador emite una
advertencia y la interfaz continúa sin icono. El icono de ventana usa los PNG
de 16, 32, 48 y 64 px; Windows intenta además `iconbitmap` con el ICO. Antes de
crear la raíz Tk, `scripts/gui.py` fija el Windows AppUserModelID
`ITBA.PSIB.LiverTumorGUI` para asociar correctamente el icono de la aplicación
con su ventana y la barra de tareas.

En Windows, `PSIB.bat` prepara el entorno, instala y verifica las dependencias, y abre la GUI. Para crear accesos `PSIB.lnk`
en el escritorio y en la raíz del proyecto, ejecutar
`powershell -NoProfile -File .\scripts\create_shortcut.ps1`. El script configura
destino, directorio de trabajo, icono y estado minimizado sin requerir permisos
de administrador. Los accesos directos son locales e ignorados por Git. En
Linux, completar `<project-root>` en `scripts/psib.desktop.template`; el campo
`Icon` apunta al PNG de 256 px.
