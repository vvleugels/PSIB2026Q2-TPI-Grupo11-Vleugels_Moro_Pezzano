# Notas de segmentación de tumores hepáticos

## Objetivo y estado

Definir el alcance y los criterios de trabajo para implementar y comparar
métodos clásicos de segmentación de tumores hepáticos. La primera etapa fijó la metodología. Las secciones finales registran las
ejecuciones exploratorias posteriores y sus resultados.

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


## Ejecución 2D en la PC local — 30 de octubre de 2026

Por la restricción de memoria de esta PC se abandonó la exploración volumétrica.
No se ejecutan componentes conectados 3D ni se cargan volúmenes completos para
la selección o segmentación. Se incorporó un lector de planos NIfTI que aplica
la orientación canónica sobre cada corte 2D.

### Selección y comandos

```bash
python scripts/select_slices_2d.py
python scripts/run_segmentation.py --run-id base
python scripts/run_segmentation.py --run-id comparacion --sweep
```

La selección 2D incluyó los 10 casos disponibles y 5 cortes por caso, con semilla
2026 y separación mínima de 2 índices: 50 cortes en total. Se prioriza área
tumoral, luego área hepática e índice. Al usar todos los casos no se aplica
estratificación por volumen ni exigencia de múltiples componentes 3D. Esto es
una adaptación explícita respecto de la selección original basada en inventario.
El CSV guardado en cada ejecución conserva el criterio y los índices exactos.

Se mantuvo fija la ventana [-140, 260] HU y la mediana de tamaño 3. Se segmentó
la imagen filtrada completa sin realce ni recorte. Las referencias tumorales
participan de la selección y la evaluación, pero no son entrada de los umbrales.

### Configuración base

La corrida `base` utiliza umbral global 40 HU (0.45 normalizado), sin apertura,
cierre, relleno ni filtro por área. Sobre los 10 casos obtuvo:

| Métrica entre casos | Media | Desvío poblacional |
| --- | ---: | ---: |
| Dice | 0.262757 | 0.254257 |
| IoU | 0.178796 | 0.189559 |

Las métricas de cada caso se calculan con conteos acumulados de sus 5 cortes;
la tabla anterior pondera los casos por igual. No es una métrica volumétrica.
Los archivos completos están en `results/segmentation/base/`.

La inspección del caso 76, corte 148, muestra que el umbral detecta una parte
importante de la referencia, pero deja borde sin cubrir y pequeños fragmentos
adicionales. Ese corte obtuvo Dice 0.766937 e IoU 0.621977. Es un ejemplo favorable,
no representativo de todos los casos.

### Estado del barrido y restricciones de ejecución

El barrido real `comparacion` quedó interrumpido y marcado como fallido; no
generó tablas completas ni configuraciones ganadoras. No se presentan resultados
comparativos de las 252 variantes. La implementación del barrido sí se verificó
con datos sintéticos, incluida su selección determinista de ganadores.

Pasaron las 34 pruebas del flujo adaptado y una prueba adicional que comprueba
que la reutilización de morfología y componentes produce las mismas máscaras
que segmentar cada configuración por separado. No se volvió a ejecutar la suite
completa que incluye el inventario volumétrico.

La restricción de esta laptop es de ejecución, no de implementación: las funciones
originales de carga volumétrica, inventario y conteo de componentes 3D permanecen
disponibles para una máquina con recursos suficientes. La selección 2D es una
alternativa y no reemplaza ni elimina ese flujo original. No hay corridas activas;
el estudio comparativo real queda pendiente de una ejecución posterior autorizada.
