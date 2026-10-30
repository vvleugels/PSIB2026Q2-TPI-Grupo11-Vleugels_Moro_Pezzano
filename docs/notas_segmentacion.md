# Notas de segmentación de tumores hepáticos

## Objetivo y estado

Definir el alcance y los criterios de trabajo para implementar y comparar
métodos clásicos de segmentación de tumores hepáticos. Esta primera etapa
documenta la metodología prevista; todavía no contiene resultados experimentales.

## Datos y alcance 2D

Se trabajará con los 10 casos locales etiquetados de `data/Selected_database/`,
reutilizando la selección de casos y cortes y el preprocesamiento existentes.
La tabla `results/selected_cases_slices.csv`, generada por el flujo de selección,
identificará los cortes utilizados. La configuración existente permite seleccionar
entre 3 y 5 cortes por caso.

Cada corte axial se procesará de manera independiente en 2D. Se conservará la
convención del proyecto: volúmenes orientados en RAS con ejes `(X, Y, Z)` y corte
axial `array[:, :, k]`. La evaluación describirá únicamente los cortes
seleccionados, sin interpretarse como una segmentación o evaluación del volumen
completo.

## Máscaras y uso de las referencias

- **Región hepática:** `seg > 0`, que incluye hígado y tumor (etiquetas 1 y 2).
  Esta región conocida delimitará dónde se buscarán los tumores.
- **Referencia tumoral:** `seg == 2`, utilizada exclusivamente para evaluar las
  predicciones en la etapa de segmentación. La selección existente de casos y
  cortes ya utiliza las referencias.
- **Entrada de los algoritmos:** la imagen del corte y su región hepática. No se
  entregará la referencia tumoral al algoritmo para calcular la predicción.

## Métodos previstos e integración

Se compararán umbral global, Otsu y umbral basado en estadísticas del hígado.
Para este último se explorará especialmente el umbral inferior, considerando
que los tumores suelen ser hipodensos. Se estudiará también el efecto de
apertura, cierre, relleno de huecos y filtrado de componentes conectados por
área mínima, con distintos elementos estructurantes.

Se conservarán las interfaces existentes:

- `psib.data_io.load_case(case_id, data_dir=None)` devuelve
  `(ct_hu, seg, spacing)`.
- `psib.dataset.select_slices(seg, n_slices, minimum_gap=5)` devuelve los índices
  de los cortes.
- `psib.preprocessing.preprocess_slice(slice_hu, liver_mask, cfg)` devuelve
  `windowed`, `denoised`, `enhanced`, `cropped` y `crop_box`.

Los módulos futuros reutilizarán estas interfaces y mantendrán la correspondencia
espacial entre las predicciones y las referencias.

## Ajuste y evaluación posterior

Los parámetros se ajustarán sobre los 10 casos locales actuales. Se registrarán
las configuraciones y sus métricas Dice e IoU, junto con observaciones sobre
aciertos y fallos, cuando se ejecuten los experimentos.

La base de datos más grande se reservará para una evaluación posterior con los
parámetros elegidos. Antes de esa evaluación se comprobará la identidad de los
casos y se excluirán los ya utilizados en el ajuste para evitar solapamiento
entre ambos conjuntos.

## Interpretación y limitaciones

Las métricas de los 10 casos serán **exploratorias**: el ajuste de parámetros y
la evaluación se realizarán sobre los mismos casos. Además, la selección de
cortes utiliza las máscaras de referencia, por lo que no constituye una muestra
prospectiva sin sesgo de selección.

La región hepática se obtiene de una máscara conocida. Por lo tanto, los
resultados describirán la búsqueda de tumores dentro de esa región y no el
desempeño de un sistema que deba detectar también el hígado desde la imagen.

Las métricas por caso resumirán solo sus cortes seleccionados. No se presentarán
como métricas de segmentación 3D ni como evidencia de generalización a casos
independientes.

## Verificación de esta etapa

El alcance, las convenciones de ejes, la selección y las interfaces se revisaron
contra el README del proyecto. Esta etapa agrega únicamente documentación:
no modifica código, configuración ni datos y no requiere ejecutar tests.
