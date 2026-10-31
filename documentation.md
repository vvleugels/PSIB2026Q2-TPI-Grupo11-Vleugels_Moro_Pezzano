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
| Cuantificación de lesiones 3D | Pendiente |
| Ejecutor de segmentación de todos los cortes y exportación de máscaras/métricas | Implementado |
| Barrido de 252 configuraciones y comparación exploratoria | Implementado |
| Comparación de filtros sobre un corte y exportación sin GUI | Implementado |
| Lanzador con fallback de datos e identificadores temporales | Implementado |
| Superposición de contornos de predicción y referencia | Disponible como API Python |
| Evaluación independiente con otra base | Pendiente |

El flujo principal usa lectura por cortes 2D para limitar el consumo de memoria.
El inventario volumétrico sigue disponible como flujo opcional. Las
características geométricas y textura GLCM 2D están implementadas; la
cuantificación 3D, compresión y reconstrucción siguen pendientes.

## 2. Estructura y responsabilidades

| Ubicación | Responsabilidad |
| --- | --- |
| `src/psib/config.py` | Lectura del YAML, precedencia local y resolución de rutas |
| `src/psib/data_io.py` | Descubrimiento, orientación, carga por volumen y lector por cortes |
| `src/psib/dataset.py` | Inventario, terciles de volumen tumoral y selección |
| `src/psib/preprocessing.py` | Ventana HU, filtros, realce y recorte hepático |
| `src/psib/segmentation.py` | Umbrales 2D, morfología y componentes conectados |
| `src/psib/metrics.py` | Dice e IoU entre máscaras binarias |
| `src/psib/features.py` | Características geométricas por lesión y textura GLCM 2D |
| `scripts/extract_features.py` | Tablas de características para cortes seleccionados |
| `scripts/plot_features.py` | Boxplots descriptivos de propiedades GLCM por tejido |
| `src/psib/visualization.py` | Contornos de predicción y referencia sobre CT |
| `scripts/run_tpi.py` | Selección 2D y segmentación base con ID temporal |
| `scripts/select_slices_2d.py` | Selección con lectura de máscaras por planos |
| `scripts/run_segmentation.py` | Ejecuciones base y barrido de parámetros |
| `scripts/compare_filters.py` | Comparación de mediana, gaussiano y bilateral |
| `scripts/explore_dataset.py` | Inventario volumétrico y figuras de contexto |
| `scripts/select_cases.py` | Selección por terciles de volumen tumoral |
| `scripts/preprocess_selected.py` | Figuras de ventana, filtrado y recorte |
| `Run_TPI.bat` | Preparación del entorno Windows y llamada al lanzador Python |
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

En Windows se puede ejecutar `Run_TPI.bat` con doble clic. Crea `.venv` si
falta y, en cada ejecución, ejecuta `pip install -r requirements.txt`
y `pip check`. Conserva los paquetes que ya satisfacen las restricciones; no
actualiza todos los paquetes a la última versión. Luego llama a
`scripts/run_tpi.py`, que usa `data_dir`
con la precedencia habitual de `config.local.yaml`. Si no encuentra pares
CT/máscara, intenta `data/Selected_database/`; si tampoco encuentra pares allí,
termina con un error. No modifica los archivos YAML del usuario.

Instalar Python 3.12 de 64 bits con el lanzador `py` para crear el entorno.
La ejecución nativa del archivo BAT no se ha verificado desde el entorno Linux.

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
| `selection.n_cases` | 10 | Casos finales |
| `selection.slices_per_case` | 5 | Cortes solicitados por caso; el script admite 3 a 5 |
| `selection.minimum_slice_gap` | 2 | Separación mínima en índices, no en mm |
| `selection.seed` | 2026 | Semilla del muestreo |
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
`compare_filters.py` y `extract_features.py`. `explore_dataset.py`, `select_cases.py` y
`preprocess_selected.py` usan el YAML de la raíz. Todos conservan la precedencia
completa de `config.local.yaml`.

`candidate_pool_size` se utiliza solo en el inventario volumétrico. La selección
2D usa `n_cases`, `slices_per_case`, `minimum_slice_gap` y `seed` sin calcular
terciles ni componentes 3D. El fallback de datos pertenece únicamente a
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
`Run_TPI.bat` prepara y verifica el entorno antes de llamar a este flujo.
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
  conteos de etiquetas, volumen tumoral y componentes tumorales 3D.
- `results/figures/hu_histogram.png`.
- `results/figures/tumor_volume_distribution.png`.
- `results/figures/spacing_distribution.png`.

El histograma HU usa hasta 100 000 vóxeles por caso del pool, muestreados en
todo el CT. El volumen tumoral en cm³ se calcula como
`tumor_voxels * sx * sy * sz / 1000`. La columna `liver_voxels` cuenta solo
la etiqueta 1, mientras que la región de segmentación `seg > 0` incluye 1 y 2.
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

Los cortes se ordenan por área tumoral descendente, luego por área hepática
y finalmente por índice. Se eligen respetando la separación mínima y se
devuelven ordenados por índice. Puede obtenerse menos cantidad que la solicitada
si el espaciado entre índices lo impide; también pueden incluirse cortes sin
tumor. La función no garantiza por sí sola que todos los cortes tengan hígado.

El CSV incluye `selection_rank`, `case_id`, `slice_index`, `tumor_volume_cm3`,
`tumor_lesions_3d`, `burden_tercile`, `candidate_pool_size`, `seed` y
`selection_criteria`.

### Figuras de preprocesamiento

`preprocess_selected.py` carga cada caso una vez por grupo de cortes. Por cada
fila del CSV crea una figura con ventana HU, imagen filtrada y recorte:

```text
results/figures/preprocess_case_003_slice_042.png
```

Los índices del nombre son ejemplos. El script exporta figuras PNG, no arrays
de preprocesamiento ni máscaras tumorales.

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
| `test_dataset.py` | Muestreo, volumen físico, componentes y selección reproducible |
| `test_preprocessing.py` | Ventana, coordenadas de recorte, forma y dtype de filtros |
| `test_compare_filters.py` | Parámetros de filtros, orientación, lectura de un corte y errores de selección |
| `test_metrics.py` | Casos idénticos, disjuntos, vacíos, solapamiento y validación |
| `test_segmentation.py` | Umbrales, región hepática, morfología, conectividad y contrato |
| `test_run_segmentation.py` | Ejecución, exportación, agregación, barrido, fallos y reproducibilidad |
| `test_run_tpi.py` | Directorio configurado, fallback, precedencia local, ID temporal y ejecución integrada |
| `test_slice_io.py` | Lectura por planos, orientación, rango de etiquetas y selección 2D |
| `test_features.py` | Geometría 2D, espaciado físico y GLCM enmascarada |

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

El estudio es 2D sobre cortes seleccionados con las referencias, dentro de
una región hepática conocida. La referencia tumoral se usa para selección
y evaluación, pero no se entrega al algoritmo de segmentación.

El ajuste previsto sobre los mismos 10 casos será exploratorio. Una métrica
por caso sobre esos cortes no equivale a evaluar el volumen completo. La
evaluación posterior en una base más grande deberá excluir los casos usados
en el ajuste y conservar la configuración elegida para esa evaluación.

El ejecutor y el barrido permiten comparar las variantes sobre los casos locales.
Queda completar el barrido experimental y realizar la evaluación independiente
con una base sin casos utilizados en el ajuste.


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
| `predictions/<configuration_id>/*.png` | Imagen filtrada, referencia y predicción, con métricas |

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

`extract_texture_features(image, roi_mask, levels=32, distances=(1,), angles=...)`
recibe una imagen 2D normalizada a `[0, 1]` y una máscara binaria de la región
de interés con la misma forma. La función cuantiza intensidades en 32 niveles
por defecto y calcula matrices simétricas para cuatro direcciones en el plano
(0°, 45°, 90° y 135°) a distancia de un píxel. Distancias, niveles y ángulos
pueden configurarse.

Solo cuenta pares donde ambos píxeles caen dentro de la región. Devuelve
contraste, homogeneidad, energía y correlación, promediados entre los
desplazamientos que contienen pares válidos. También devuelve `pair_count` y
`n_offsets`. Si ningún desplazamiento contiene pares, las cuatro propiedades
son NaN y ambos conteos son cero. Para comparar textura de las referencias,
usar por separado `seg == 2` para tumor y `seg == 1` para hígado no tumoral, junto
con la imagen filtrada normalizada.

Estas características resumen una región en un corte; sus comparaciones son
descriptivas cuando la región se define con máscaras de referencia. La API
permite analizar tanto una referencia como una predicción, pero esta entrega no
debe interpretarse como evaluación independiente. El resumen generado es
descriptivo y no realiza pruebas de hipótesis.

### Tablas por selección

Desde la raíz del repositorio y después de generar `<output_dir>/selected_cases_slices.csv`:

```bash
python scripts/extract_features.py
python scripts/plot_features.py
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

`pair_count` cuenta los pares espaciales válidos antes de simetrizar cada GLCM;
`n_offsets` cuenta las combinaciones distancia/ángulo que aportaron pares. Las
propiedades sin pares válidos son NaN. En la tabla de resumen, `n_slices` ignora
esos valores NaN para cada propiedad. Las máscaras de referencia definen ambas
regiones, por lo que la comparación de tejidos es descriptiva. Para las
diferencias por caso, la prueba exacta bilateral de signos evalúa si hay una
dirección consistente (los empates se excluyen del contraste), y Holm corrige
las cuatro comparaciones GLCM. Es un análisis exploratorio con pocos casos
seleccionados, no una validación independiente de la segmentación.

El segundo comando lee ambas tablas de textura y guarda dos figuras, cada una
con cuatro paneles: contraste, homogeneidad, energía y correlación. En la figura
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
describe magnitud y dirección entre los pares disponibles, sin prueba de
significancia ni afirmación de separación generalizable.

El comando acepta `--config ruta/a/config.yaml`. Usa el `data_dir` configurado
sin aplicar el fallback del lanzador; corregir esa ruta si fue necesario el fallback.
